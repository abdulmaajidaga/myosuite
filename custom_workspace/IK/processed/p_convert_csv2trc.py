import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import resample, butter, filtfilt

# ========================================
# CONFIGURATION
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # IK/processed/
IK_DIR = os.path.dirname(BASE_DIR)  # IK/
PROJECT_ROOT = os.path.dirname(IK_DIR)  # custom_workspace/

# 1. Locations of ORIGINAL RAW files (Source of Time/Frames)
RAW_DIRS = [
    os.path.join(PROJECT_ROOT, "data/kinematic/Healthy"),
    os.path.join(PROJECT_ROOT, "data/kinematic/Stroke")
]

# 2. Locations of PROCESSED files (Source of Chest-Relative Data + Vectors)
# Usually inside a 'processed' subfolder, or flattened.
# This script searches recursively to find them.
PROCESSED_ROOT_DIRS = RAW_DIRS

# 3. Output (in IK/output, not IK/processed/output)
OUTPUT_DIR = os.path.join(IK_DIR, "output/trc/p_originals_w_vector")

DATA_RATE = 200.0  # MHH/Vicon Standard
CUTOFF_FREQ = 6.0  # Biomechanical smoothing

# ========================================

def get_original_frame_count(raw_filepath):
    """ Reads raw file just to get the true length (rows). """
    try:
        # Skip 2 header rows, count data
        df = pd.read_csv(raw_filepath, header=None, skiprows=2, usecols=[0])
        return len(df)
    except: 
        return None

def load_processed_data(filepath):
    """ Loads the 100-frame normalized data """
    try:
        df = pd.read_csv(filepath)
        if 'WrVec_x' not in df.columns: return None
        return df
    except: return None

def resample_to_original(df_proc, target_frames):
    """ 
    Stretches the 100 frames back to Target Frames (e.g. 2124) 
    to restore the original speed of movement.
    """
    curr_frames = len(df_proc)
    if curr_frames == target_frames: return df_proc
    
    print(f"   -> Restoring Time: {curr_frames} frames -> {target_frames} frames")
    
    new_data = {}
    for col in df_proc.columns:
        # Scipy resample handles the interpolation perfectly
        new_data[col] = resample(df_proc[col].values, target_frames)
        
    return pd.DataFrame(new_data, columns=df_proc.columns)

def filter_data(df, cutoff=6, fs=200):
    """ Smooths the resampled data """
    nyq = 0.5 * fs
    b, a = butter(4, cutoff/nyq, btype='low')
    
    filt = df.copy()
    cols = [c for c in df.columns if 'Vec' not in c] # Don't filter vectors (optional)
    
    for c in cols:
        filt[c] = filtfilt(b, a, df[c])
    return filt

def format_4_markers(df):
    """
    Creates the 4 Virtual Markers for OpenSim.
    1. V_Shoulder
    2. V_Elbow
    3. V_Wrist (Center)
    4. V_Pinky (Calculated from Vector for Rotation)
    """
    trc = pd.DataFrame()
    
    # Standard
    trc['V_Shoulder_X'] = df['Sh_x']
    trc['V_Shoulder_Y'] = df['Sh_y']
    trc['V_Shoulder_Z'] = df['Sh_z']
    
    trc['V_Elbow_X']    = df['El_x']
    trc_data_y          = df['El_y'] # temp var
    trc['V_Elbow_Y']    = df['El_y']
    trc['V_Elbow_Z']    = df['El_z']
    
    trc['V_Wrist_X']    = df['Wr_x']
    trc['V_Wrist_Y']    = df['Wr_y']
    trc['V_Wrist_Z']    = df['Wr_z']

    # V_PINKY Calculation
    # WrVec was (Pinky - Thumb). 
    # V_Pinky = Center + 0.5 * Vector
    trc['V_Vector_X'] = df['Wr_x'] + (df['WrVec_x'] * 0.5)
    trc['V_Vector_Y'] = df['Wr_y'] + (df['WrVec_y'] * 0.5)
    trc['V_Vector_Z'] = df['Wr_z'] + (df['WrVec_z'] * 0.5)
    
    return trc

def save_trc(df, filepath, fs=200):
    markers = ['V_Shoulder', 'V_Elbow', 'V_Wrist', 'V_Vector']
    n_frames = len(df)
    n_markers = len(markers)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(filepath)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fs}\t{fs}\t{n_frames}\t{n_markers}\tmm\t{fs}\t1\t{n_frames}\n")
        f.write("Frame#\tTime\t" + "\t".join([f"{m}\t\t" for m in markers]) + "\n")
        f.write("\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(n_markers)]) + "\n")
        f.write("\n")
        
    # Prepare data block
    out = pd.DataFrame()
    out['Frame#'] = range(1, n_frames + 1)
    out['Time'] = out['Frame#'] / fs
    
    for m in markers:
        out[f'{m}_X'] = df[f'{m}_X']
        out[f'{m}_Y'] = df[f'{m}_Y']
        out[f'{m}_Z'] = df[f'{m}_Z']
        
    out.to_csv(filepath, sep='\t', index=False, header=False, mode='a', lineterminator='\n')
    print(f"✓ Saved: {os.path.basename(filepath)} ({n_frames} frames)")

def main():
    print("--- TRC Conversion: Restoring Original Time & Rotation ---")
    
    processed_files_cache = {}
    # Pre-scan processed files to make lookup faster
    print("Scanning processed files...")
    for d in PROCESSED_ROOT_DIRS:
        if os.path.exists(d):
            # Find all _processed.csv
            found = glob.glob(os.path.join(d, "**", "*_processed.csv"), recursive=True)
            for f in found:
                # Store as { 'S1_12_1': 'path/to/S1_12_1_processed.csv' }
                key = os.path.basename(f).replace('_processed.csv', '')
                processed_files_cache[key] = f
                
    print(f"Found {len(processed_files_cache)} processed files.")
    
    # Iterate Raw Files
    for raw_dir in RAW_DIRS:
        if not os.path.exists(raw_dir): continue
        
        raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
        print(f"\nProcessing directory: {raw_dir}")
        
        for raw_f in raw_files:
            if "_processed" in raw_f: continue # Skip if processed file is in raw dir
            
            base_name = os.path.basename(raw_f).replace('.csv', '')
            
            # 1. Match with Processed
            if base_name not in processed_files_cache:
                # print(f"Skipping {base_name} (No processed file found)")
                continue
                
            proc_path = processed_files_cache[base_name]
            
            # 2. Get Time from Raw
            n_frames = get_original_frame_count(raw_f)
            if not n_frames: continue
            
            # 3. Get Data from Processed
            df_proc = load_processed_data(proc_path)
            if df_proc is None: continue
            
            # 4. Resample
            df_resampled = resample_to_original(df_proc, n_frames)
            
            # 5. Filter
                # df_filt = filter_data(df_resampled, CUTOFF_FREQ, DATA_RATE)
            df_filt = df_resampled  # No filtering for now
            # 6. Format (Add Pinky)
            df_trc = format_4_markers(df_filt)
            
            # 7. Save
            out_name = os.path.join(OUTPUT_DIR, f"{base_name}.trc")
            save_trc(df_trc, out_name, DATA_RATE)

if __name__ == "__main__":
    main()