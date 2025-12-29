"""
===============================================================================
FILE: viz_utils.py
===============================================================================
Shared utilities for data loading, path management, and configuration.
"""

import os
import numpy as np
import pandas as pd
from scipy import signal
import glob

# --- CONFIGURATION ---
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
IK_DIR = os.path.join(BASE_DIR, "IK")
OUTPUT_DIR = os.path.join(IK_DIR, "visual")
DATA_DIR = os.path.join(IK_DIR, "output")

STROKE_DIR = os.path.join(BASE_DIR, "data/kinematic/Stroke")
HEALTHY_DIR = os.path.join(BASE_DIR, "data/kinematic/Healthy")

SCORES_FILE = os.path.join(DATA_DIR, "scores.csv")
MODEL_PATH = os.path.join(DATA_DIR, "motion_generator_model.pkl")
MODEL_XML = os.path.join(BASE_DIR, "model/myo_sim/arm/myoarm.xml")

FIXED_FRAMES = 100

def get_mot_files(directory):
    """Returns list of .mot files in directory."""
    return sorted(glob.glob(os.path.join(directory, "*.mot")))

def read_mot_file(filepath, resample=True):
    """Parses a MOT file and returns data (flattened if resample=True, else DataFrame)."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        header_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == 'endheader':
                header_idx = i
                break
        
        col_names = lines[header_idx+1].strip().split('\t')
        data = np.loadtxt(lines[header_idx+2:])
        
        if resample:
            joints = data[:, 1:] # Drop time
            return signal.resample(joints, FIXED_FRAMES).flatten()
        else:
            return pd.DataFrame(data, columns=col_names)
            
    except Exception as e:
        # print(f"Error reading MOT {filepath}: {e}")
        return None

def read_trc_file(filepath):
    """Parses a TRC file and returns flattened resampled data."""
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        
        # Find header
        header_line = 3
        for i, line in enumerate(lines):
            if "Frame#" in line:
                header_line = i
                break
        
        start_line = header_line + 2
        data_block = []
        for line in lines[start_line:]:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                try:
                    # Skip Frame# (0) and Time (1)
                    vals = [float(x) for x in parts[2:] if x != '']
                    data_block.append(vals)
                except ValueError: pass

        if not data_block: return None
        data_np = np.array(data_block)
        return signal.resample(data_np, FIXED_FRAMES).flatten()
        
    except Exception as e:
        # print(f"Error reading TRC {filepath}: {e}")
        return None

def read_csv_file(filepath):
    """Parses MHH CSV with multi-level header and returns flattened resampled data."""
    try:
        # Load data (skip header rows)
        df = pd.read_csv(filepath, skiprows=2, header=None)
        
        # Ensure consistent number of columns (63)
        if df.shape[1] > 63:
            df = df.iloc[:, :63]
        elif df.shape[1] < 63:
            return None

        # Simple imputation
        data_clean = df.ffill().bfill().fillna(0).values
        
        # Resample
        return signal.resample(data_clean, FIXED_FRAMES).flatten()
    except Exception as e:
        # print(f"Error reading CSV {filepath}: {e}")
        return None
