import numpy as np
import pandas as pd
from scipy import signal
import os
import random
import glob
import matplotlib
matplotlib.use('Agg') 
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

# Target Columns in Order
FINAL_COLS = [
    'Sh_x','Sh_y','Sh_z',
    'El_x','El_y','El_z',
    'Wr_x','Wr_y','Wr_z',
    'WrVec_x','WrVec_y','WrVec_z'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helpers ---

def load_and_validate(filepath):
    """
    Loads CSV and ensures the required skeleton columns exist.
    """
    df = pd.read_csv(filepath)
    
    # Check if we have the specific headers we need
    # We do a case-insensitive check and rename to standard if needed
    lower_cols = {c.lower(): c for c in df.columns}
    
    # Mapping for standardizing internal names
    standard_map = {}
    missing = []
    
    for target in FINAL_COLS:
        # Check if target exists (case insensitive)
        if target.lower() in lower_cols:
            standard_map[lower_cols[target.lower()]] = target
        else:
            missing.append(target)
            
    if missing:
        # If WrVec is missing, we can tolerate it and fill with zeros or compute it
        # But Sh, El, Wr are critical.
        critical = [m for m in missing if 'Vec' not in m]
        if critical:
            print(f"   [!] Skipping {os.path.basename(filepath)}: Missing columns {critical}")
            return None
    
    # Rename columns to standard format
    df = df.rename(columns=standard_map)
    
    # Fill missing WrVec with zeros if necessary (or compute Wr-El later if desired)
    for m in missing:
        df[m] = 0.0
        
    return df[FINAL_COLS] # Return only relevant columns in correct order

def check_and_fix_units(df_stroke, df_healthy):
    """
    Fixes mm vs m mismatch.
    """
    # Check Wrist Range
    range_s = (df_stroke[['Wr_x','Wr_y','Wr_z']].max() - df_stroke[['Wr_x','Wr_y','Wr_z']].min()).mean()
    range_h = (df_healthy[['Wr_x','Wr_y','Wr_z']].max() - df_healthy[['Wr_x','Wr_y','Wr_z']].min()).mean()
    
    if range_s > (range_h * 50):
        print("   -> Scaling Stroke units (mm -> m)")
        return df_stroke / 1000.0, df_healthy
    elif range_h > (range_s * 50):
        print("   -> Scaling Healthy units (mm -> m)")
        return df_stroke, df_healthy / 1000.0
        
    return df_stroke, df_healthy

def resample_dataframe(df, target_len):
    """Resamples all columns to target length."""
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_len)
    return pd.DataFrame(new_data, columns=df.columns)

# --- Core Logic: Skeleton Relativization ---

def get_relative_skeleton(df):
    """
    Converts global coordinates to Shoulder-Relative coordinates.
    Returns: 
      - shoulder_track (The original shoulder global positions)
      - relative_df (El, Wr, WrVec relative to 0,0,0 shoulder)
    """
    shoulder_cols = ['Sh_x', 'Sh_y', 'Sh_z']
    elbow_cols    = ['El_x', 'El_y', 'El_z']
    wrist_cols    = ['Wr_x', 'Wr_y', 'Wr_z']
    vec_cols      = ['WrVec_x', 'WrVec_y', 'WrVec_z']
    
    shoulder_track = df[shoulder_cols].copy()
    
    # Create relative dataframe
    rel_df = pd.DataFrame(index=df.index)
    
    # Elbow relative to Shoulder
    rel_df['El_x'] = df['El_x'] - df['Sh_x']
    rel_df['El_y'] = df['El_y'] - df['Sh_y']
    rel_df['El_z'] = df['El_z'] - df['Sh_z']
    
    # Wrist relative to Shoulder (Standard for FMA reach)
    # Alternatively, Wrist relative to Elbow? 
    # For FMA, calculating both relative to Shoulder is safer for pure trajectory morphing.
    rel_df['Wr_x'] = df['Wr_x'] - df['Sh_x']
    rel_df['Wr_y'] = df['Wr_y'] - df['Sh_y']
    rel_df['Wr_z'] = df['Wr_z'] - df['Sh_z']
    
    # Vectors are usually directions, so they don't need position subtraction
    # Just copy them
    rel_df['WrVec_x'] = df['WrVec_x']
    rel_df['WrVec_y'] = df['WrVec_y']
    rel_df['WrVec_z'] = df['WrVec_z']
    
    return shoulder_track, rel_df

def reconstruct_skeleton(shoulder_track, relative_df):
    """
    Adds the morphed relative movements back to the Locked Shoulder track.
    """
    out_df = pd.DataFrame(index=relative_df.index)
    
    # 1. Shoulder (Locked/Original)
    out_df['Sh_x'] = shoulder_track['Sh_x']
    out_df['Sh_y'] = shoulder_track['Sh_y']
    out_df['Sh_z'] = shoulder_track['Sh_z']
    
    # 2. Elbow (Shoulder + Relative Elbow)
    out_df['El_x'] = out_df['Sh_x'] + relative_df['El_x']
    out_df['El_y'] = out_df['Sh_y'] + relative_df['El_y']
    out_df['El_z'] = out_df['Sh_z'] + relative_df['El_z']
    
    # 3. Wrist (Shoulder + Relative Wrist)
    out_df['Wr_x'] = out_df['Sh_x'] + relative_df['Wr_x']
    out_df['Wr_y'] = out_df['Sh_y'] + relative_df['Wr_y']
    out_df['Wr_z'] = out_df['Sh_z'] + relative_df['Wr_z']
    
    # 4. Vectors (Passed through)
    out_df['WrVec_x'] = relative_df['WrVec_x']
    out_df['WrVec_y'] = relative_df['WrVec_y']
    out_df['WrVec_z'] = relative_df['WrVec_z']
    
    return out_df[FINAL_COLS]

# --- Morphing ---

def morph_chain(df_stroke, df_healthy, target_score, start_score):
    """
    Morphs the Elbow/Wrist/Vec relative to shoulder, while keeping shoulder locked.
    """
    # 1. Decompose Stroke (Keep Shoulder, Get Relative Arm)
    s_shoulder, s_rel = get_relative_skeleton(df_stroke)
    
    # 2. Decompose Healthy (Get Relative Arm only)
    # We ignore healthy shoulder position entirely
    _, h_rel = get_relative_skeleton(df_healthy)
    
    # 3. Align Healthy Relative to Stroke Relative Length
    target_len = len(df_healthy) # Or len(df_stroke) depending on speed preference
    # Usually we want the speed of healthy, so we resize stroke to healthy length
    s_rel_aligned = resample_dataframe(s_rel, target_len)
    s_shoulder_aligned = resample_dataframe(s_shoulder, target_len)
    
    # 4. Interpolate
    alpha = (target_score - start_score) / (HIGH_FMA - start_score)
    
    # Morph the relative arm positions
    morphed_rel = (1 - alpha) * s_rel_aligned + alpha * h_rel
    
    # Add Noise (Tremor simulation on the arm only)
    noise_scale = 0.005 * (1 - alpha) # Small noise
    noise = np.random.normal(0, noise_scale, morphed_rel.shape)
    morphed_rel += noise
    
    # 5. Reconstruct (Attach new arm to Old Shoulder)
    final_df = reconstruct_skeleton(s_shoulder_aligned, morphed_rel)
    
    return final_df

# --- Animation ---

def animate_skeleton(df_stroke, df_healthy, generated_snapshots, start_score, save_path):
    """
    Animates the full Arm (Sh-El-Wr) recovery.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Resample source for visualization consistency
    viz_len = len(df_healthy)
    s_viz = resample_dataframe(df_stroke, viz_len)
    
    # Calc limits
    all_x = pd.concat([s_viz['Wr_x'], df_healthy['Wr_x']])
    all_y = pd.concat([s_viz['Wr_y'], df_healthy['Wr_y']])
    all_z = pd.concat([s_viz['Wr_z'], df_healthy['Wr_z']])
    
    # Padding
    pad = 0.2
    ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
    ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
    ax.set_zlim(all_z.min()-pad, all_z.max()+pad)
    
    def update(score):
        ax.clear()
        ax.set_xlim(all_x.min()-pad, all_x.max()+pad)
        ax.set_ylim(all_y.min()-pad, all_y.max()+pad)
        ax.set_zlim(all_z.min()-pad, all_z.max()+pad)
        ax.set_title(f"Skeleton Morph: FMA {score}")
        
        # Helper to draw arm
        def plot_arm(df, color, alpha, style='-'):
            # Draw Shoulder -> Elbow -> Wrist
            ax.plot([df['Sh_x'].mean(), df['El_x'].mean()], 
                    [df['Sh_y'].mean(), df['El_y'].mean()], 
                    [df['Sh_z'].mean(), df['El_z'].mean()], c=color, alpha=alpha, linestyle=style)
            ax.plot([df['El_x'].mean(), df['Wr_x'].mean()], 
                    [df['El_y'].mean(), df['Wr_y'].mean()], 
                    [df['El_z'].mean(), df['Wr_z'].mean()], c=color, alpha=alpha, linestyle=style)
            # Draw Trajectory
            ax.plot(df['Wr_x'], df['Wr_y'], df['Wr_z'], c=color, alpha=alpha/2, linewidth=1)

        # 1. Original Stroke (Red)
        plot_arm(s_viz, 'red', 0.2, '--')
        
        # 2. Target Healthy (Green) - Shifted to match stroke shoulder for viz comparison
        # (Optional: Shift healthy to stroke shoulder just for visual comparison)
        h_shifted = df_healthy.copy()
        diff_x = s_viz['Sh_x'].mean() - df_healthy['Sh_x'].mean()
        diff_y = s_viz['Sh_y'].mean() - df_healthy['Sh_y'].mean()
        diff_z = s_viz['Sh_z'].mean() - df_healthy['Sh_z'].mean()
        h_shifted['Wr_x'] += diff_x; h_shifted['Wr_y'] += diff_y; h_shifted['Wr_z'] += diff_z
        h_shifted['El_x'] += diff_x; h_shifted['El_y'] += diff_y; h_shifted['El_z'] += diff_z
        h_shifted['Sh_x'] += diff_x; h_shifted['Sh_y'] += diff_y; h_shifted['Sh_z'] += diff_z
        plot_arm(h_shifted, 'green', 0.2, '--')

        # 3. Current Morph (Blue)
        if score in generated_snapshots:
            df_curr = generated_snapshots[score]
            # Draw Bones (Last Frame Position)
            last = df_curr.iloc[-1]
            ax.plot([last['Sh_x'], last['El_x']], [last['Sh_y'], last['El_y']], [last['Sh_z'], last['El_z']], 
                    c='blue', linewidth=3, marker='o')
            ax.plot([last['El_x'], last['Wr_x']], [last['El_y'], last['Wr_y']], [last['El_z'], last['Wr_z']], 
                    c='blue', linewidth=3, marker='o')
            # Draw Trajectory
            ax.plot(df_curr['Wr_x'], df_curr['Wr_y'], df_curr['Wr_z'], c='blue', alpha=0.6)

    scores = sorted(generated_snapshots.keys())
    anim_scores = scores[::2]
    anim = FuncAnimation(fig, update, frames=anim_scores, interval=100)
    try:
        anim.save(save_path, writer=PillowWriter(fps=15))
    except: pass
    plt.close(fig)

# --- Main ---

def batch_process():
    print("--- Skeleton Morphing Started ---")
    
    # Load Scores
    try:
        df_scores = pd.read_csv(SCORES_FILE)
        id_col, score_col = df_scores.columns[0], df_scores.columns[1]
        df_scores[id_col] = df_scores[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        score_map = dict(zip(df_scores[id_col], df_scores[score_col]))
    except: print("Error loading scores."); return

    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
    
    print(f"Files: {len(stroke_files)} Stroke, {len(healthy_files)} Healthy")

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('_processed.csv', '').replace('.csv', '')
        start_score = score_map.get(s_name)
        if not start_score: 
            for k in score_map: 
                if k in s_name: start_score = score_map[k]; break
        
        if not start_score or int(start_score) >= 60: continue
        start_score = int(start_score)
        
        print(f"Processing {s_name} (FMA {start_score})...")
        
        df_s = load_and_validate(s_file)
        if df_s is None: continue

        partners = random.sample(healthy_files, min(len(healthy_files), PAIRS_PER_STROKE))
        
        for h_file in partners:
            h_name = os.path.basename(h_file).split('.')[0]
            df_h = load_and_validate(h_file)
            if df_h is None: continue

            # Unit fix only (No mirroring)
            df_s_fixed, df_h_fixed = check_and_fix_units(df_s, df_h)
            
            pair_dir = os.path.join(OUTPUT_DIR, f"{s_name}_to_{h_name}")
            if not os.path.exists(pair_dir): os.makedirs(pair_dir)
            
            snapshots = {}
            for score in range(start_score + 1, HIGH_FMA):
                df_gen = morph_chain(df_s_fixed, df_h_fixed, score, start_score)
                df_gen.to_csv(os.path.join(pair_dir, f"FMA_{score}.csv"), index=False)
                snapshots[score] = df_gen
            
            # Animation
            animate_skeleton(df_s_fixed, df_h_fixed, snapshots, start_score, os.path.join(pair_dir, "skeleton_anim.gif"))

    print("Done.")

if __name__ == "__main__":
    batch_process()