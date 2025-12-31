import os
import glob
import pandas as pd
import numpy as np
import sys
from scipy.signal import resample

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Assumes IK is at root/IK
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "kinematic")
OUTPUT_SUBDIR = "processed"

# Define required marker sets
MARKER_GROUPS = {
    'Chest': ['CS_1', 'CS_2', 'CS_3', 'CS_4'],
    'Shoulder': ['SA_1', 'SA_2', 'SA_3'],
    'Elbow': ['ELB_L', 'ELB_M'],
    'Wrist': ['WRA', 'WRB']
}

OUTPUT_COLUMNS = [
    'Sh_x', 'Sh_y', 'Sh_z',
    'El_x', 'El_y', 'El_z',
    'Wr_x', 'Wr_y', 'Wr_z',
    'WrVec_x', 'WrVec_y', 'WrVec_z'
]

def find_marker_indices(header_row):
    """
    Parses the header row (list of strings) and returns a dict mapping
    MarkerName -> ColumnIndex (start of the triplet).
    """
    marker_map = {}
    for idx, val in enumerate(header_row):
        if pd.notna(val) and str(val).strip() != "":
            marker_name = str(val).strip()
            marker_map[marker_name] = idx
    return marker_map

def get_marker_data(df_data, start_idx):
    """
    Extracts 3 columns (X,Y,Z) starting at start_idx.
    Returns (N, 3) numpy array.
    """
    # Slice columns idx, idx+1, idx+2
    # Ensure numeric
    try:
        data = df_data.iloc[:, start_idx:start_idx+3].apply(pd.to_numeric, errors='coerce')
        # Fill missing within trajectory
        data = data.ffill().bfill().fillna(0.0)
        return data.values
    except Exception as e:
        print(f"    Error extracting data at index {start_idx}: {e}")
        return None

def process_file(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    
    try:
        # Read Header separate from data
        # Row 0: Marker Names
        # Row 1: X,Y,Z (ignore)
        # Row 2+: Data
        
        # Read just header first
        df_header = pd.read_csv(file_path, header=None, nrows=1)
        header_row = df_header.iloc[0].tolist()
        marker_map = find_marker_indices(header_row)
        
        # Check for missing markers
        missing_markers = []
        for group, markers in MARKER_GROUPS.items():
            for m in markers:
                if m not in marker_map:
                    missing_markers.append(m)
        
        if missing_markers:
            print(f"  Skipping. Missing markers: {missing_markers}")
            return

        # Read Data
        # Skip 2 rows (Marker Names, XYZ)
        df_data = pd.read_csv(file_path, header=None, skiprows=2)
        
        # Calculate Joint Centers
        joint_centers = {}
        
        for group, markers in MARKER_GROUPS.items():
            group_data = []
            for m in markers:
                idx = marker_map[m]
                xyz = get_marker_data(df_data, idx)
                if xyz is None:
                    print(f"  Skipping. Failed to read data for {m}")
                    return
                group_data.append(xyz)
            
            # Stack and Average
            # group_data is list of (N, 3)
            # Average across markers (axis 0 of the stacked array)
            stacked = np.array(group_data) # (M, N, 3)
            center = np.mean(stacked, axis=0) # (N, 3)
            joint_centers[group] = center
            
        # Get individual wrist markers for vector
        wra = get_marker_data(df_data, marker_map['WRA'])
        wrb = get_marker_data(df_data, marker_map['WRB'])
        
        # Calculate Wrist Vector (WRB - WRA)
        wrist_vector = wrb - wra # (N, 3)
        
        # Normalize (Subtract Chest Center)
        chest_c = joint_centers['Chest']
        
        sh_norm = joint_centers['Shoulder'] - chest_c
        el_norm = joint_centers['Elbow'] - chest_c
        wr_norm = joint_centers['Wrist'] - chest_c
        # Wrist Vector is NOT normalized by position
        
        # Combine Output
        # [Sh, El, Wr, WrVec]
        output_data = np.concatenate([sh_norm, el_norm, wr_norm, wrist_vector], axis=1) # (N, 12)
        
        # Resample to 100 frames
        output_data = resample(output_data, 100) # (100, 12)
        
        # Create DataFrame
        df_out = pd.DataFrame(output_data, columns=OUTPUT_COLUMNS)
        
        # Save
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        processed_dir = os.path.join(dir_name, OUTPUT_SUBDIR)
        os.makedirs(processed_dir, exist_ok=True)
        
        out_name = f"{base_name}_processed.csv"
        out_path = os.path.join(processed_dir, out_name)
        
        df_out.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")
        
    except Exception as e:
        print(f"  Error processing file: {e}")

def main():
    # Process both directories
    dirs = [
        os.path.join(DATA_DIR, "Healthy"),
        os.path.join(DATA_DIR, "Stroke")
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            continue
            
        csv_files = glob.glob(os.path.join(d, "*.csv"))
        print(f"\nFound {len(csv_files)} files in {d}")
        
        for f in csv_files:
            process_file(f)

if __name__ == "__main__":
    main()