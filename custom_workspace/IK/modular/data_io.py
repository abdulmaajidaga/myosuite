"""
Functions for reading and writing motion data files (TRC, MOT)
"""
import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Union, Tuple

def read_mot_file(filepath: str) -> Optional[pd.DataFrame]:
    """Read a .mot file and return as DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MOT file not found: {filepath}")
    
    skiprows = 0
    with open(filepath, "r") as file:
        for line in file:
            if "endheader" in line:
                break
            skiprows += 1
    
    return pd.read_csv(filepath, sep=r'\s+', skiprows=skiprows + 1)

def write_mot_file(filepath: str, joint_trajectories: np.ndarray, 
                   joint_names: List[str], time: np.ndarray) -> None:
    """Write joint trajectories to .mot file."""
    num_frames = len(time)
    with open(filepath, 'w') as f:
        f.write(f"{os.path.basename(filepath)}\nversion=1\n")
        f.write(f"nRows={num_frames}\nnColumns={len(joint_names) + 1}\n")
        f.write("inDegrees=no\nendheader\n")
        f.write("time\t" + "\t".join(joint_names) + "\n")
        
        for i in range(num_frames):
            row = [f"{time[i]:.8f}"] + [f"{q:.8f}" for q in joint_trajectories[i]]
            f.write("\t".join(row) + "\n")

def read_trc_file(filepath: str, scale_to_meters: bool = True) -> Tuple[Dict[str, np.ndarray], float]:
    """Read marker trajectories and metadata from TRC file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    rate_line = lines[2].strip().split('\t')
    data_rate = float(rate_line[0])
    
    # Find column headers
    marker_line = None
    data_start = None
    for i, line in enumerate(lines):
        if 'Frame#\tTime' in line:
            marker_line = i
            data_start = i + 2  # Skip coordinate labels line
            break
    
    if marker_line is None:
        raise ValueError("Could not find marker headers in TRC file")
        
    # Parse marker names
    markers = lines[marker_line].strip().split('\t')[2:]
    markers = [m.strip() for m in markers if m.strip()]
    
    # Read data using pandas for efficiency
    data = pd.read_csv(filepath, sep='\t', skiprows=data_start, header=None)
    
    # Extract marker trajectories
    trajectories = {}
    scale = 0.001 if scale_to_meters else 1.0
    for i, marker in enumerate(markers):
        cols = [2 + i*3, 3 + i*3, 4 + i*3]  # X,Y,Z columns for this marker
        trajectories[marker] = data[cols].values * scale
        
    return trajectories, data_rate

def write_trc_file(filepath: str, marker_trajectories: Dict[str, np.ndarray], 
                   data_rate: float = 100.0, units: str = 'mm') -> None:
    """Write marker trajectories to TRC file."""
    markers = list(marker_trajectories.keys())
    num_frames = len(next(iter(marker_trajectories.values())))
    
    with open(filepath, 'w') as f:
        # Write header
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate}\t{data_rate}\t{num_frames}\t{len(markers)}\t{units}\t{data_rate}\t1\t{num_frames}\n")
        
        # Write column headers
        f.write("Frame#\tTime\t" + "\t".join([f"{marker}\t\t" for marker in markers]) + "\n")
        f.write("\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(len(markers))]) + "\n")
        f.write("\n")
        
        # Write data
        time = np.arange(num_frames) / data_rate
        for i in range(num_frames):
            row = [f"{i+1}", f"{time[i]:.3f}"]
            for marker in markers:
                row.extend([f"{x:.6f}" for x in marker_trajectories[marker][i]])
            f.write("\t".join(row) + "\n")