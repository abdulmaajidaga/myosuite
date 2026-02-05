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

# 1. Where are the ORIGINAL RAW files? (To get the time/frames)
RAW_DIRS = [
    os.path.join(PROJECT_ROOT, "data/kinematic/Healthy"),
    os.path.join(PROJECT_ROOT, "data/kinematic/Stroke")
]

# 2. Where are the PROCESSED files? (To get the clean data)
# (Assumes they are in a 'processed' subfolder inside the raw dirs)
PROCESSED_SUBDIR = "processed"

# 3. Where should the TRC files go? (in IK/output, not IK/processed/output)
OUTPUT_DIR = os.path.join(IK_DIR, "output/trc/p_originals")

# Sampling Rate of your Camera System (Vicon/OptiTrack)
# Change this if your camera is 100Hz or 120Hz
DATA_RATE = 200.0 

# Filtering (Optional smoothing)
APPLY_FILTER = True
CUTOFF_FREQ = 6.0

# ========================================

def get_original_frame_count(raw_filepath):
    """
    Reads the raw CSV just to count the number of data frames.
    Skipping the 2 header rows standard in your MHH format.
    """
    try:
        # Read only the first column to be fast
        # Skip 2 header rows
        df = pd.read_csv(raw_filepath, header=None, skiprows=2, usecols=[0])
        return len(df)
    except Exception as e:
        print(f"Error reading raw file {raw_filepath}: {e}")
        return None

def resample_to_original(df_processed, target_frames):
    """
    Resamples the 100-frame processed data back to N-frames.
    """
    current_frames = len(df_processed)
    if current_frames == target_frames:
        return df_processed
    
    # Resample
    print(f"   -> Resampling: {current_frames} frames -> {target_frames} frames")
    new_data = {}
    
    for col in df_processed.columns:
        # Scipy resample works on the array
        new_data[col] = resample(df_processed[col].values, target_frames)
        
    return pd.DataFrame(new_data, columns=df_processed.columns)

def filter_data(df, cutoff, fs):
    """ Standard Low-Pass Filter """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    
    filtered_df = df.copy()
    # Filter only position columns, skip any text or headers
    cols = [c for c in df.columns if 'Vec' not in c] # Standard columns
    
    for col in cols:
        filtered_df[col] = filtfilt(b, a, df[col])
        
    return filtered_df

def format_and_save_trc(df, filename, output_dir, fs):
    """ Save to OpenSim TRC format """
    # Map to Virtual Markers
    trc_data = pd.DataFrame()
    trc_data['V_Shoulder_X'] = df['Sh_x']
    trc_data['V_Shoulder_Y'] = df['Sh_y']
    trc_data['V_Shoulder_Z'] = df['Sh_z']
    
    trc_data['V_Elbow_X']    = df['El_x']
    trc_data['V_Elbow_Y']    = df['El_y']
    trc_data['V_Elbow_Z']    = df['El_z']
    
    trc_data['V_Wrist_X']    = df['Wr_x']
    trc_data['V_Wrist_Y']    = df['Wr_y']
    trc_data['V_Wrist_Z']    = df['Wr_z']
    
    markers = ['V_Shoulder', 'V_Elbow', 'V_Wrist']
    num_markers = len(markers)
    num_frames = len(trc_data)
    
    # Create Output Directory
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename.replace('.csv', '.trc'))
    
    # Write File
    with open(out_path, 'w') as f:
        # Header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(out_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fs}\t{fs}\t{num_frames}\t{num_markers}\tmm\t{fs}\t1\t{num_frames}\n")
        
        # Column Headers
        header1 = "Frame#\tTime\t" + "\t".join([f"{m}\t\t" for m in markers])
        f.write(header1 + "\n")
        header2 = "\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
        f.write(header2 + "\n")
        f.write("\n")
        
    # Prepare Data Block
    out_block = pd.DataFrame()
    out_block['Frame#'] = range(1, num_frames + 1)
    out_block['Time'] = out_block['Frame#'] / fs
    
    for col in trc_data.columns:
        out_block[col] = trc_data[col]
        
    out_block.to_csv(out_path, sep='\t', index=False, header=False, mode='a', lineterminator='\n')
    print(f"   ✓ Saved: {out_path} ({num_frames/fs:.2f}s duration)")

def main():
    print("--- Restoring Original Timeline for TRC Conversion ---")
    
    for raw_dir in RAW_DIRS:
        if not os.path.exists(raw_dir):
            print(f"Skipping {raw_dir} (Not found)")
            continue
            
        print(f"\nScanning: {raw_dir}")
        
        # Find all RAW CSVs
        raw_files = glob.glob(os.path.join(raw_dir, "*.csv"))
        
        for raw_file in raw_files:
            # Check if this is a raw file (and not a processed one mistakenly placed here)
            if "_processed" in raw_file: continue
            
            base_name = os.path.basename(raw_file).replace('.csv', '')
            
            # 1. Get Original Time (Frames)
            orig_frames = get_original_frame_count(raw_file)
            if not orig_frames: continue
            
            # 2. Find Corresponding Processed File
            # Assuming standard location: raw_dir/processed/filename_processed.csv
            processed_path = os.path.join(raw_dir, PROCESSED_SUBDIR, f"{base_name}_processed.csv")
            
            if not os.path.exists(processed_path):
                print(f"   x Missing processed file for: {base_name}")
                continue
                
            # 3. Load Processed Data (100 frames)
            try:
                df_proc = pd.read_csv(processed_path)
            except:
                print(f"   x Error reading processed file: {base_name}")
                continue
            
            # 4. Resample back to Original
            df_restored = resample_to_original(df_proc, orig_frames)
            
            # 5. Filter (Optional)
            if APPLY_FILTER:
                df_restored = filter_data(df_restored, CUTOFF_FREQ, DATA_RATE)
            
            # 6. Save TRC
            format_and_save_trc(df_restored, f"{base_name}.trc", OUTPUT_DIR, DATA_RATE)

if __name__ == "__main__":
    main()