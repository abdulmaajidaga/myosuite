import numpy as np
import pandas as pd
from scipy import signal
import os
import random
import glob
import matplotlib
matplotlib.use('Agg') # Headless mode for batch processing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
OUTPUT_DIR = os.path.join(root_data, "data/kinematic/augmented")
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

HIGH_FMA = 66
PAIRS_PER_STROKE = 2 
REFERENCE_KEYWORDS = ["sh", "shoulder", "acromion"] # Columns to subtract (Origin)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Standardization Helpers ---

def identify_columns(df):
    """Separates Shoulder (Reference) and Hand (Limb) columns."""
    cols = df.columns
    ref_cols = []
    limb_cols = []
    for col in cols:
        if any(k in col.lower() for k in REFERENCE_KEYWORDS):
            ref_cols.append(col)
        elif np.issubdtype(df[col].dtype, np.number):
            limb_cols.append(col)
    return ref_cols, limb_cols

def normalize_origin(df):
    """
    1. Subtracts Shoulder from Hand (isolates arm reach).
    2. Translates to start at (0,0,0).
    """
    ref_cols, limb_cols = identify_columns(df)
    
    if len(ref_cols) < 3 or len(limb_cols) < 3:
        # Fallback: Just subtract the first row (start position)
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df - numeric_df.iloc[0]

    ref_cols.sort()
    limb_cols.sort()
    
    df_rel = pd.DataFrame(index=df.index)
    valid_dims = 3
    final_col_names = ['X', 'Y', 'Z'] 
    
    for i in range(valid_dims):
        df_rel[final_col_names[i]] = df[limb_cols[i]] - df[ref_cols[i]]
        
    start_pos = df_rel.iloc[0].values
    return df_rel - start_pos

def standardize_units_and_direction(df_stroke, df_healthy):
    """
    1. Fixes mm vs m mismatch.
    2. Mirrors Stroke data if it moves in the opposite direction to Healthy (Left vs Right arm).
    """
    # 1. Unit Scaling (mm to m)
    range_s = (df_stroke.max() - df_stroke.min()).mean()
    range_h = (df_healthy.max() - df_healthy.min()).mean()
    
    if range_s > (range_h * 50):
        print("   -> Scaling Stroke units (mm -> m)")
        df_stroke = df_stroke / 1000.0
    elif range_h > (range_s * 50):
        print("   -> Scaling Healthy units (mm -> m)")
        df_healthy = df_healthy / 1000.0
        
    # 2. Directional Mirroring (Left Arm vs Right Arm)
    # We calculate the primary vector of movement (End - Start)
    vec_s = df_stroke.iloc[-1] - df_stroke.iloc[0]
    vec_h = df_healthy.iloc[-1] - df_healthy.iloc[0]
    
    # Check X-axis (usually Left/Right)
    if np.sign(vec_s['X']) != np.sign(vec_h['X']):
        # If one moves Left (-X) and other Right (+X), flip Stroke X
        print("   -> Mirroring Stroke X-axis (Left/Right Arm correction)")
        df_stroke['X'] = df_stroke['X'] * -1

    return df_stroke, df_healthy

# --- Morphing Logic ---

def resample_path(df, target_length):
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_length)
    return pd.DataFrame(new_data)

def generate_morph(df_stroke, df_healthy, target_score, start_score):
    alpha = (target_score - start_score) / (HIGH_FMA - start_score)
    df_stroke_aligned = resample_path(df_stroke, len(df_healthy))
    
    # Linear Interpolation
    df_new = (1 - alpha) * df_stroke_aligned + alpha * df_healthy
    
    # Noise (decreases as score improves)
    noise_scale = 0.01 * (1 - alpha) 
    noise = np.random.normal(0, noise_scale, df_new.shape)
    
    return df_new + noise

# --- Animation & Visualization ---

def create_morph_animation(df_stroke, df_healthy, generated_snapshots, start_score, save_path):
    """
    Creates a GIF showing the trajectory Evolving from Stroke -> Healthy.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Data prep
    cols = ['X', 'Y', 'Z']
    df_s_aligned = resample_path(df_stroke, len(df_healthy))
    
    # Determine fixed axis limits so the camera doesn't jump
    all_data = pd.concat([df_s_aligned, df_healthy] + list(generated_snapshots.values()))
    x_min, x_max = all_data['X'].min(), all_data['X'].max()
    y_min, y_max = all_data['Y'].min(), all_data['Y'].max()
    z_min, z_max = all_data['Z'].min(), all_data['Z'].max()

    def update(score):
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_title(f"Recovery Simulation: FMA Score {score}")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        
        # Static: Stroke (Red Ghost) and Healthy (Green Ghost)
        ax.plot(df_s_aligned['X'], df_s_aligned['Y'], df_s_aligned['Z'], 
                c='red', alpha=0.1, linewidth=1)
        ax.plot(df_healthy['X'], df_healthy['Y'], df_healthy['Z'], 
                c='green', alpha=0.1, linewidth=1)
        
        # Active: The current morph (Blue)
        if score in generated_snapshots:
            current_df = generated_snapshots[score]
            # Color shifts from Red -> Blue -> Green
            alpha_val = (score - start_score) / (HIGH_FMA - start_score)
            color_blend = (1-alpha_val, 0, alpha_val) # RGB blending
            
            ax.plot(current_df['X'], current_df['Y'], current_df['Z'], 
                    c=color_blend, linewidth=3, label=f"FMA {score}")
            
            # Draw a dot at the end position
            ax.scatter(current_df['X'].iloc[-1], current_df['Y'].iloc[-1], current_df['Z'].iloc[-1], 
                       c='black', s=20)

    # Create frames for available scores
    scores = sorted(generated_snapshots.keys())
    # Downsample frames for GIF speed (e.g. every 2nd score)
    anim_scores = scores[::2] 
    
    anim = FuncAnimation(fig, update, frames=anim_scores, interval=100)
    
    # Save
    try:
        anim.save(save_path, writer=PillowWriter(fps=10))
    except Exception as e:
        print(f"   [!] Animation save failed (missing ffmpeg/imagemagick?): {e}")
    plt.close(fig)

# --- Main Processor ---

def batch_process():
    print(f"--- FMA Generator: Standardized & Animated ---")
    
    # Load Score Map
    try:
        df_scores = pd.read_csv(SCORES_FILE)
        id_col, score_col = df_scores.columns[0], df_scores.columns[1]
        df_scores[id_col] = df_scores[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        score_map = dict(zip(df_scores[id_col], df_scores[score_col]))
    except Exception as e:
        print(f"Error loading scores: {e}"); return

    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
    
    print(f"Found {len(stroke_files)} Stroke and {len(healthy_files)} Healthy files.")

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('_processed.csv', '').replace('.csv', '')
        
        # Score Lookup
        start_score = score_map.get(s_name)
        if not start_score:
            for k in score_map: 
                if k in s_name: start_score = score_map[k]; break
        
        if not start_score or int(start_score) >= 60: continue
        start_score = int(start_score)
        print(f"\nProcessing {s_name} (FMA {start_score}) ...")

        # 1. Load & Normalize Origin (Shoulder Ref)
        df_s_raw = pd.read_csv(s_file)
        df_s_norm = normalize_origin(df_s_raw)

        partners = random.sample(healthy_files, min(len(healthy_files), PAIRS_PER_STROKE))
        
        for h_file in partners:
            h_name = os.path.basename(h_file).split('.')[0]
            df_h_raw = pd.read_csv(h_file)
            df_h_norm = normalize_origin(df_h_raw)
            
            # 2. Standardize Units & Direction (Mirroring)
            df_s_final, df_h_final = standardize_units_and_direction(df_s_norm, df_h_norm)
            
            # Setup Output
            pair_id = f"{s_name}_to_{h_name}"
            pair_dir = os.path.join(OUTPUT_DIR, pair_id)
            if not os.path.exists(pair_dir): os.makedirs(pair_dir)
            
            snapshots = {}
            error_log = []

            # 3. Generate Loop
            for score in range(start_score + 1, HIGH_FMA):
                df_gen = generate_morph(df_s_final, df_h_final, score, start_score)
                df_gen.to_csv(os.path.join(pair_dir, f"FMA_{score}.csv"), index=False)
                
                # Store for animation
                snapshots[score] = df_gen
                
                # Error Log
                rmse = np.sqrt(((df_gen.values - df_h_final.values)**2).mean())
                error_log.append({'FMA': score, 'RMSE': rmse})

            # Save Analytics
            pd.DataFrame(error_log).to_csv(os.path.join(pair_dir, "error_analysis.csv"), index=False)
            
            # 4. Generate 3D Animation
            anim_path = os.path.join(pair_dir, "morph_animation.gif")
            create_morph_animation(df_s_final, df_h_final, snapshots, start_score, anim_path)
            
            print(f" -> Generated {len(snapshots)} files. Animation saved: {anim_path}")

if __name__ == "__main__":
    batch_process()