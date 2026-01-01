import numpy as np
import pandas as pd
from scipy import signal
import os
import random
import glob
import matplotlib
# Use 'Agg' backend to prevent crashes on servers or when generating many plots
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

# Update these paths if your folder structure differs slightly
STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
OUTPUT_DIR = os.path.join(root_data, "data/kinematic/augmented")
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

HIGH_FMA = 66
PAIRS_PER_STROKE = 2 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helpers ---
def load_scores_map(filepath):
    """
    Reads scores.csv and returns a dictionary: {'filename_without_ext': score}
    """
    try:
        df = pd.read_csv(filepath)
        id_col = df.columns[0]
        score_col = df.columns[1] 
        
        # Clean IDs: remove .mot, strip whitespace
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        score_map = dict(zip(df[id_col], df[score_col]))
        return score_map
    except Exception as e:
        print(f"CRITICAL ERROR loading scores.csv: {e}")
        return {}

def load_csv_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure all data is numeric, drop any non-numeric columns
    df = df.select_dtypes(include=[np.number])
    return df

def resample_path(df, target_length):
    new_data = {}
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            new_data[col] = signal.resample(df[col], target_length)
    return pd.DataFrame(new_data)

def calculate_rmse(df_gen, df_target):
    # Align columns just in case
    common_cols = df_gen.columns.intersection(df_target.columns)
    diff = df_gen[common_cols].values - df_target[common_cols].values
    squared_diff = diff ** 2
    return np.sqrt(squared_diff.mean())

# --- Visualization Function ---
def plot_comparison_3d(df_stroke, df_healthy, generated_snapshots, start_score, save_path):
    """
    Generates a 3D plot comparing the original stroke movement, 
    the target healthy movement, and the generated intermediates.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Heuristic: Use columns 0, 1, 2 for X, Y, Z. 
    # If you have 'Time' as col 0, change this to [1, 2, 3]
    cols = df_stroke.columns[:3]
    
    # 1. Plot Original Stroke (RED)
    # Resample temporarily for plotting alignment
    df_s_aligned = resample_path(df_stroke, len(df_healthy))
    ax.plot(df_s_aligned[cols[0]], df_s_aligned[cols[1]], df_s_aligned[cols[2]], 
            c='red', label=f'Start: FMA {start_score}', linewidth=3, alpha=0.7)

    # 2. Plot Generated Intermediates (BLUE Gradients)
    sorted_scores = sorted(generated_snapshots.keys())
    for score in sorted_scores:
        df_gen = generated_snapshots[score]
        
        # Calculate fade: Score 20 = Light Blue, Score 60 = Dark Blue
        rel_alpha = (score - start_score) / (HIGH_FMA - start_score)
        # Ensure alpha is between 0.1 and 1
        plot_alpha = max(0.1, min(1.0, rel_alpha))
        
        ax.plot(df_gen[cols[0]], df_gen[cols[1]], df_gen[cols[2]], 
                c='blue', alpha=0.3, linewidth=1)

    # 3. Plot Target Healthy (GREEN)
    ax.plot(df_healthy[cols[0]], df_healthy[cols[1]], df_healthy[cols[2]], 
            c='green', label='Target: FMA 66', linewidth=3)

    # Labels and Legend
    ax.set_title(f"Kinematic Reconstruction: FMA {start_score} $\u2192$ 66")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.legend()
    
    # Save and Close
    plt.savefig(save_path)
    plt.close(fig) # Crucial to free memory in loops

# --- Core Morphing Logic ---
def generate_morph(df_stroke, df_healthy, target_score, start_score):
    # Dynamic Alpha: Where does target_score sit between start_score and 66?
    alpha = (target_score - start_score) / (HIGH_FMA - start_score)
    
    target_len = len(df_healthy)
    df_stroke_aligned = resample_path(df_stroke, target_len)
    
    # Linear Interpolation
    df_new = (1 - alpha) * df_stroke_aligned + alpha * df_healthy
    
    # Add variable noise (Tremor Simulation)
    # Noise is high when alpha is low, noise is 0 when alpha is 1
    noise_scale = 0.02 * (1 - alpha) 
    noise = np.random.normal(0, noise_scale, df_new.shape)
    
    return df_new + noise

# --- Main Processor ---
def batch_process():
    print(f"--- FMA Generator Initialized ---")
    print(f"Loading scores from: {SCORES_FILE}")
    
    score_map = load_scores_map(SCORES_FILE)
    if not score_map:
        print("ERROR: Could not load score map. Checking path...")
        print(f"Expected path: {SCORES_FILE}")
        return

    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
    
    print(f"Found {len(stroke_files)} Stroke files and {len(healthy_files)} Healthy files.")

    for s_file in stroke_files:
        # Match filename to score ID
        s_name_raw = os.path.basename(s_file)
        s_name = s_name_raw.replace('_processed.csv', '').replace('.csv', '')
        
        # Try exact match, then substring match
        start_score = score_map.get(s_name)
        if start_score is None:
            # Fallback: search keys
            for key in score_map:
                if key in s_name: 
                    start_score = score_map[key]
                    break
        
        if start_score is None:
            print(f"Skipping {s_name}: FMA score not found.")
            continue
            
        start_score = int(start_score)
        
        # Don't generate if patient is already near 66
        if start_score >= 60:
            print(f"Skipping {s_name}: FMA {start_score} is too high to generate meaningful data.")
            continue

        print(f"\nProcessing {s_name} (FMA {start_score}) ...")

        df_stroke = load_csv_data(s_file)
        
        # Pick random Healthy partners
        partners = random.sample(healthy_files, min(len(healthy_files), PAIRS_PER_STROKE))
        
        for h_file in partners:
            h_name = os.path.basename(h_file).split('.')[0]
            df_healthy = load_csv_data(h_file)
            
            pair_id = f"{s_name}_to_{h_name}"
            pair_dir = os.path.join(OUTPUT_DIR, pair_id)
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)
            
            snapshots_for_plot = {}
            error_log = []

            # Loop: Generate Score + 1 up to 65
            for score in range(start_score + 1, HIGH_FMA):
                df_generated = generate_morph(df_stroke, df_healthy, score, start_score)
                
                out_name = f"FMA_{score}.csv"
                df_generated.to_csv(os.path.join(pair_dir, out_name), index=False)
                
                rmse = calculate_rmse(df_generated, df_healthy)
                error_log.append({'FMA': score, 'RMSE': rmse})
                
                # Save snapshots for visualization (every 10 points or min/max)
                if score % 10 == 0 or score == start_score + 1:
                    snapshots_for_plot[score] = df_generated

            # Save Analytics
            pd.DataFrame(error_log).to_csv(os.path.join(pair_dir, "error_analysis.csv"), index=False)
            
            # Save Visualization
            plot_path = os.path.join(pair_dir, "trajectory_viz.png")
            plot_comparison_3d(df_stroke, df_healthy, snapshots_for_plot, start_score, plot_path)
            
            print(f" -> Pair Done. Viz saved to: {plot_path}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    batch_process()