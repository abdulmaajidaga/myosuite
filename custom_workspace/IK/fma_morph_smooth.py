import numpy as np
import pandas as pd
from scipy import signal
import os
import random
import glob

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
OUTPUT_DIR = os.path.join(root_data, "data/kinematic/augmented_smooth") # New Folder
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

HIGH_FMA = 66
PAIRS_PER_STROKE = 2 

# Standard Columns
FINAL_COLS = [
    'Sh_x','Sh_y','Sh_z',
    'El_x','El_y','El_z',
    'Wr_x','Wr_y','Wr_z',
    'WrVec_x','WrVec_y','WrVec_z'
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. Signal Processing Helpers ---

def butter_lowpass_filter(data, cutoff=6, fs=100, order=2):
    """
    Standard biomechanical filter to remove jitter/noise.
    cutoff: 6Hz is standard for voluntary human movement.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def apply_smoothing(df):
    """Applies filter to all coordinate columns."""
    df_clean = df.copy()
    for col in df.columns:
        # Only smooth actual coordinates, not vectors if they are unit vectors
        # But for simplicity, we smooth everything to ensure continuity
        df_clean[col] = butter_lowpass_filter(df[col].values)
    return df_clean

# --- 2. Data Loading & Logic ---

def force_meters(df):
    """
    Aggressively forces units to meters.
    If the average value is > 10, it's definitely millimeters -> divide by 1000.
    """
    # Check Wrist X range or mean
    check_val = (df['Wr_x'].max() - df['Wr_x'].min())
    if check_val > 50: # If movement range is > 50 units, it's mm (50m reach is impossible)
        return df / 1000.0
    return df

def load_and_prep(filepath):
    try:
        df = pd.read_csv(filepath)
    except: return None
    
    # Standardize Column Names
    lower_cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    for target in FINAL_COLS:
        if target.lower() in lower_cols:
            rename_map[lower_cols[target.lower()]] = target
    
    df = df.rename(columns=rename_map)
    
    # Ensure all columns exist
    for c in FINAL_COLS:
        if c not in df.columns: df[c] = 0.0
            
    df = df[FINAL_COLS]
    
    # FORCE UNITS TO METERS
    df = force_meters(df)
    
    return df

def resample_dataframe(df, target_len):
    new_data = {}
    for col in df.columns:
        new_data[col] = signal.resample(df[col], target_len)
    return pd.DataFrame(new_data, columns=df.columns)

def get_relative_skeleton(df):
    shoulder_track = df[['Sh_x', 'Sh_y', 'Sh_z']].copy()
    rel_df = pd.DataFrame(index=df.index)
    
    # Calculate Relative Positions (Limb - Shoulder)
    for part in ['El', 'Wr']:
        rel_df[f'{part}_x'] = df[f'{part}_x'] - df['Sh_x']
        rel_df[f'{part}_y'] = df[f'{part}_y'] - df['Sh_y']
        rel_df[f'{part}_z'] = df[f'{part}_z'] - df['Sh_z']
    
    # Vectors just copy
    rel_df['WrVec_x'] = df['WrVec_x']
    rel_df['WrVec_y'] = df['WrVec_y']
    rel_df['WrVec_z'] = df['WrVec_z']
    
    return shoulder_track, rel_df

def reconstruct_skeleton(shoulder_track, relative_df):
    out_df = pd.DataFrame(index=relative_df.index)
    
    # Shoulder
    out_df['Sh_x'] = shoulder_track['Sh_x']
    out_df['Sh_y'] = shoulder_track['Sh_y']
    out_df['Sh_z'] = shoulder_track['Sh_z']
    
    # Limbs
    for part in ['El', 'Wr']:
        out_df[f'{part}_x'] = out_df['Sh_x'] + relative_df[f'{part}_x']
        out_df[f'{part}_y'] = out_df['Sh_y'] + relative_df[f'{part}_y']
        out_df[f'{part}_z'] = out_df['Sh_z'] + relative_df[f'{part}_z']
        
    # Vectors
    out_df['WrVec_x'] = relative_df['WrVec_x']
    out_df['WrVec_y'] = relative_df['WrVec_y']
    out_df['WrVec_z'] = relative_df['WrVec_z']
    
    return out_df[FINAL_COLS]

# --- 3. The Smoothed Morph ---

def morph_chain_smooth(df_stroke, df_healthy, target_score, start_score):
    alpha = (target_score - start_score) / (HIGH_FMA - start_score)
    
    # Temporal Morph
    len_s, len_h = len(df_stroke), len(df_healthy)
    target_len = int((1 - alpha) * len_s + alpha * len_h)
    
    s_sh, s_rel = get_relative_skeleton(df_stroke)
    _, h_rel = get_relative_skeleton(df_healthy)
    
    # Resample
    s_rel_aln = resample_dataframe(s_rel, target_len)
    h_rel_aln = resample_dataframe(h_rel, target_len)
    s_sh_aln = resample_dataframe(s_sh, target_len)
    
    # Spatial Morph (NO RANDOM NOISE ADDED)
    morphed_rel = (1 - alpha) * s_rel_aln + alpha * h_rel_aln
    
    # Reconstruct
    raw_morph = reconstruct_skeleton(s_sh_aln, morphed_rel)
    
    # APPLY SMOOTHING FINAL STEP
    final_df = apply_smoothing(raw_morph)
    
    return final_df

# --- 4. Batch Processor ---

def main():
    print("--- Starting Smoothed FMA Generation ---")
    try:
        score_map = pd.read_csv(SCORES_FILE)
        score_map = dict(zip(
            score_map.iloc[:,0].astype(str).str.replace('.mot','',regex=False).str.strip(),
            score_map.iloc[:,1]
        ))
    except: return

    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))

    for s_file in stroke_files:
        s_name = os.path.basename(s_file).replace('_processed.csv','').replace('.csv','')
        
        # Get Score
        start_score = score_map.get(s_name)
        if not start_score:
            for k in score_map: 
                if k in s_name: start_score = score_map[k]; break
        if not start_score or int(start_score) >= 60: continue
        start_score = int(start_score)
        
        print(f"Processing {s_name} (FMA {start_score})...")
        
        df_s = load_and_prep(s_file)
        if df_s is None: continue

        partners = random.sample(healthy_files, min(len(healthy_files), PAIRS_PER_STROKE))
        
        for h_file in partners:
            h_name = os.path.basename(h_file).split('.')[0]
            df_h = load_and_prep(h_file)
            if df_h is None: continue
            
            pair_dir = os.path.join(OUTPUT_DIR, f"{s_name}_to_{h_name}")
            if not os.path.exists(pair_dir): os.makedirs(pair_dir)
            
            for score in range(start_score + 1, HIGH_FMA):
                df_gen = morph_chain_smooth(df_s, df_h, score, start_score)
                df_gen.to_csv(os.path.join(pair_dir, f"FMA_{score}.csv"), index=False)
                
    print(f"Done! Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()