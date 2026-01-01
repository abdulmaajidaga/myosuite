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
from matplotlib.animation import FuncAnimation, PillowWriter

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

# Update these paths to match your folder structure exactly
STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
OUTPUT_DIR = os.path.join(root_data, "data/kinematic/augmented")
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

HIGH_FMA = 66
PAIRS_PER_STROKE = 2 

# The exact columns you requested
FINAL_COLS = [
    'Sh_x','Sh_y','Sh_z',
    'El_x','El_y','El_z',
    'Wr_x','Wr_y','Wr_z',
    'WrVec_x','WrVec_y','WrVec_z'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. Data Loading & Standardization Helpers ---

def load_scores_map(filepath):
    """
    Reads scores.csv and maps ID -> Score.
    """
    try:
        df = pd.read_csv(filepath)
        id_col = df.columns[0]
        score_col = df.columns[1] 
        # Clean IDs: remove .mot, strip whitespace to match filenames
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        score_map = dict(zip(df[id_col], df[score_col]))
        return score_map
    except Exception as e:
        print(f"CRITICAL ERROR loading scores.csv: {e}")
        return {}

def load_and_validate(filepath):
    """
    Loads CSV, looks for required columns, and renames them to standard names.
    Returns DataFrame with exactly 12 columns.
    """
    try:
        df = pd.read_csv(filepath)
    except:
        return None
        
    # Create a mapping from existing columns to target columns
    # We check case-insensitively (e.g., 'sh_x' matches 'Sh_x')
    lower_cols = {c.lower(): c for c in df.columns}
    
    standard_map = {}
    missing = []
    
    for target in FINAL_COLS:
        if target.lower() in lower_cols:
            standard_map[lower_cols[target.lower()]] = target
        else:
            missing.append(target)
            
    # If critical columns (Joints) are missing, skip this file
    # We can tolerate missing Vectors (WrVec) and fill them with 0
    critical_missing = [m for m in missing if 'Vec' not in m]
    if critical_missing:
        # print(f"   [!] Skipping {os.path.basename(filepath)}: Missing {critical_missing}")
        return None
    
    # Rename found columns
    df = df.rename(columns=standard_map)
    
    # Fill missing non-critical columns with 0
    for m in missing:
        df[m] = 0.0
        
    return df[FINAL_COLS]

def check_and_fix_units(df_stroke, df_healthy):
    """
    Detects if one file is in mm and the other in m. Scales to match.
    """
    # Compare the physical range of movement of the Wrist
    range_s = (df_stroke[['Wr_x','Wr_y','Wr_z']].max() - df_stroke[['Wr_x','Wr_y','Wr_z']].min()).mean()
    range_h = (df_healthy[['Wr_x','Wr_y','Wr_z']].max() - df_healthy[['Wr_x','Wr_y','Wr_z']].min()).mean()
    
    # Heuristic: If one range is > 50x the other, it's likely mm vs m
    if range_s > (range_h * 50):
        print("   -> Unit Mismatch: Scaling Stroke down (mm -> m)")
        return df_stroke / 1000.0, df_healthy
    elif range_h > (range_s * 50):
        print("   -> Unit Mismatch: Scaling Healthy down (mm -> m)")
        return df_stroke, df_healthy / 1000.0
        
    return df_stroke, df_healthy

def resample_dataframe(df, target_len):
    """Resamples all columns to a specific number of frames (Time Morphing)."""
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_len)
    return pd.DataFrame(new_data, columns=df.columns)

# --- 2. Skeleton Logic (Relative Morphing) ---

def get_relative_skeleton(df):
    """
    Decomposes the skeleton.
    Returns:
    1. shoulder_track: The global position of the shoulder (to be locked).
    2. rel_df: The Elbow and Wrist positions RELATIVE to the Shoulder.
    """
    shoulder_track = df[['Sh_x', 'Sh_y', 'Sh_z']].copy()
    
    rel_df = pd.DataFrame(index=df.index)
    
    # Elbow relative to Shoulder
    rel_df['El_x'] = df['El_x'] - df['Sh_x']
    rel_df['El_y'] = df['El_y'] - df['Sh_y']
    rel_df['El_z'] = df['El_z'] - df['Sh_z']
    
    # Wrist relative to Shoulder
    rel_df['Wr_x'] = df['Wr_x'] - df['Sh_x']
    rel_df['Wr_y'] = df['Wr_y'] - df['Sh_y']
    rel_df['Wr_z'] = df['Wr_z'] - df['Sh_z']
    
    # Vectors (Directions don't need translation, just copy)
    rel_df['WrVec_x'] = df['WrVec_x']
    rel_df['WrVec_y'] = df['WrVec_y']
    rel_df['WrVec_z'] = df['WrVec_z']
    
    return shoulder_track, rel_df

def reconstruct_skeleton(shoulder_track, relative_df):
    """
    Rebuilds the global skeleton by attaching the morphed arm 
    back to the original (locked) shoulder track.
    """
    out_df = pd.DataFrame(index=relative_df.index)
    
    # 1. Shoulder (Locked)
    out_df['Sh_x'] = shoulder_track['Sh_x']
    out_df['Sh_y'] = shoulder_track['Sh_y']
    out_df['Sh_z'] = shoulder_track['Sh_z']
    
    # 2. Elbow (Shoulder + Relative)
    out_df['El_x'] = out_df['Sh_x'] + relative_df['El_x']
    out_df['El_y'] = out_df['Sh_y'] + relative_df['El_y']
    out_df['El_z'] = out_df['Sh_z'] + relative_df['El_z']
    
    # 3. Wrist (Shoulder + Relative)
    out_df['Wr_x'] = out_df['Sh_x'] + relative_df['Wr_x']
    out_df['Wr_y'] = out_df['Sh_y'] + relative_df['Wr_y']
    out_df['Wr_z'] = out_df['Sh_z'] + relative_df['Wr_z']
    
    # 4. Vectors
    out_df['WrVec_x'] = relative_df['WrVec_x']
    out_df['WrVec_y'] = relative_df['WrVec_y']
    out_df['WrVec_z'] = relative_df['WrVec_z']
    
    return out_df[FINAL_COLS]

def morph_chain(df_stroke, df_healthy, target_score, start_score):
    """
    The Core Algorithm.
    Morphs both SPACE (Skeleton Shape) and TIME (Speed/Duration).
    """
    # 1. Calculate Morph Factor (Alpha)
    alpha = (target_score - start_score) / (HIGH_FMA - start_score)
    
    # 2. Temporal Morphing (Calculate new duration)
    # This gradually increases speed from Stroke-Duration to Healthy-Duration
    len_s = len(df_stroke)
    len_h = len(df_healthy)
    target_len = int((1 - alpha) * len_s + alpha * len_h)
    
    # 3. Decompose Skeletons
    s_shoulder, s_rel = get_relative_skeleton(df_stroke)
    _, h_rel = get_relative_skeleton(df_healthy) # Ignore healthy shoulder
    
    # 4. Resample to Target Time
    # We stretch/squeeze both the Stroke arm and the Healthy arm to the new "intermediate" speed
    s_rel_aligned = resample_dataframe(s_rel, target_len)
    h_rel_aligned = resample_dataframe(h_rel, target_len)
    
    # Also resample the Stroke Shoulder (Locked Anchor) to match the new time
    s_shoulder_aligned = resample_dataframe(s_shoulder, target_len)
    
    # 5. Spatial Morphing
    morphed_rel = (1 - alpha) * s_rel_aligned + alpha * h_rel_aligned
    
    # 6. Noise Injection (Tremor Simulation)
    # Noise fades as alpha approaches 1 (Healthy)
    noise_scale = 0.005 * (1 - alpha)
    noise = np.random.normal(0, noise_scale, morphed_rel.shape)
    morphed_rel += noise
    
    # 7. Reconstruct
    final_df = reconstruct_skeleton(s_shoulder_aligned, morphed_rel)
    
    return final_df

# --- 3. Animation Generator ---

def animate_skeleton(df_stroke, df_healthy, generated_snapshots, start_score, save_path):
    """
    Creates a GIF showing the Stroke Arm (Red) morphing into the Generated Arm (Blue).
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Healthy length for the static visual container
    viz_len = len(df_healthy)
    s_viz = resample_dataframe(df_stroke, viz_len)
    
    # Determine Axis Limits
    all_x = pd.concat([s_viz['Wr_x'], df_healthy['Wr_x']])
    all_y = pd.concat([s_viz['Wr_y'], df_healthy['Wr_y']])
    all_z = pd.concat([s_viz['Wr_z'], df_healthy['Wr_z']])
    pad = 0.2
    
    def update(score):
        ax.clear()
        ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
        ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
        ax.set_zlim(all_z.min()-pad, all_z.max()+pad)
        ax.set_title(f"FMA {score} (Locked Shoulder)")
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # Helper to draw bones
        def plot_bones(df, color, style, alpha):
            # Sh->El
            ax.plot([df['Sh_x'].mean(), df['El_x'].mean()], 
                    [df['Sh_y'].mean(), df['El_y'].mean()], 
                    [df['Sh_z'].mean(), df['El_z'].mean()], 
                    c=color, linestyle=style, alpha=alpha, linewidth=2)
            # El->Wr
            ax.plot([df['El_x'].mean(), df['Wr_x'].mean()], 
                    [df['El_y'].mean(), df['Wr_y'].mean()], 
                    [df['El_z'].mean(), df['Wr_z'].mean()], 
                    c=color, linestyle=style, alpha=alpha, linewidth=2)

        # Draw Static References (Ghosts)
        plot_bones(s_viz, 'red', '--', 0.2) # Stroke Ghost
        
        # Shift Healthy to Stroke Shoulder for visual comparison only
        h_shift = df_healthy.copy()
        diff = s_viz.iloc[0][['Sh_x','Sh_y','Sh_z']].values - df_healthy.iloc[0][['Sh_x','Sh_y','Sh_z']].values
        for part in ['Sh','El','Wr']:
            h_shift[f'{part}_x'] += diff[0]
            h_shift[f'{part}_y'] += diff[1]
            h_shift[f'{part}_z'] += diff[2]
        plot_bones(h_shift, 'green', '--', 0.2) # Healthy Ghost

        # Draw Active Morph (Blue)
        if score in generated_snapshots:
            curr = generated_snapshots[score]
            # Draw Path
            ax.plot(curr['Wr_x'], curr['Wr_y'], curr['Wr_z'], c='blue', alpha=0.6, linewidth=1)
            # Draw Bones at final frame position
            last = curr.iloc[-1]
            ax.plot([last['Sh_x'], last['El_x']], [last['Sh_y'], last['El_y']], [last['Sh_z'], last['El_z']], c='blue', linewidth=3, marker='o')
            ax.plot([last['El_x'], last['Wr_x']], [last['El_y'], last['Wr_y']], [last['El_z'], last['Wr_z']], c='blue', linewidth=3, marker='o')

    scores = sorted(generated_snapshots.keys())
    # Subsample frames for GIF speed (every 2nd score)
    anim_scores = scores[::2] 
    
    anim = FuncAnimation(fig, update, frames=anim_scores, interval=150)
    try:
        anim.save(save_path, writer=PillowWriter(fps=10))
    except Exception as e:
        print(f"Warning: GIF save failed: {e}")
    plt.close(fig)

# --- 4. Main Processing Loop ---

def batch_process():
    print("--- Starting Skeleton Morphing Process ---")
    
    # 1. Load Scores
    score_map = load_scores_map(SCORES_FILE)
    if not score_map:
        print("Stopping: Score map empty or file not found.")
        return

    # 2. Get Files
    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
    
    print(f"Found {len(stroke_files)} Stroke files and {len(healthy_files)} Healthy files.")

    for s_file in stroke_files:
        s_name_raw = os.path.basename(s_file)
        # Clean filename to match ID in scores.csv
        s_name = s_name_raw.replace('_processed.csv', '').replace('.csv', '')
        
        # 3. Find Start Score
        start_score = score_map.get(s_name)
        if start_score is None:
            # Try fuzzy match
            for k in score_map:
                if k in s_name:
                    start_score = score_map[k]
                    break
        
        if start_score is None:
            print(f"Skipping {s_name}: FMA score not found.")
            continue
            
        start_score = int(start_score)
        
        # Only process if there is room to improve
        if start_score >= 60:
            continue
            
        print(f"Processing {s_name} (FMA {start_score})...")
        
        # 4. Load Stroke Data
        df_s = load_and_validate(s_file)
        if df_s is None: continue # Skip if missing columns

        # 5. Pick Random Healthy Partners
        partners = random.sample(healthy_files, min(len(healthy_files), PAIRS_PER_STROKE))
        
        for h_file in partners:
            h_name = os.path.basename(h_file).split('.')[0]
            df_h = load_and_validate(h_file)
            if df_h is None: continue

            # 6. Unit Check (mm vs m)
            df_s_fixed, df_h_fixed = check_and_fix_units(df_s, df_h)
            
            # Prepare Output Folder
            pair_id = f"{s_name}_to_{h_name}"
            pair_dir = os.path.join(OUTPUT_DIR, pair_id)
            if not os.path.exists(pair_dir):
                os.makedirs(pair_dir)
            
            snapshots = {}
            
            # 7. Generate Loop (Start+1 -> 65)
            for score in range(start_score + 1, HIGH_FMA):
                # Run the Morph (Time + Space)
                df_gen = morph_chain(df_s_fixed, df_h_fixed, score, start_score)
                
                # Save CSV
                out_name = f"FMA_{score}.csv"
                df_gen.to_csv(os.path.join(pair_dir, out_name), index=False)
                
                # Store for animation
                snapshots[score] = df_gen

            # 8. Create Visualization
            anim_path = os.path.join(pair_dir, "morph_animation.gif")
            animate_skeleton(df_s_fixed, df_h_fixed, snapshots, start_score, anim_path)
            
    print("\nBatch processing complete.")

if __name__ == "__main__":
    batch_process()