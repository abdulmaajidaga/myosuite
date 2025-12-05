"""
Functions for processing kinematic data, filtering, and computing virtual markers
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import Dict, Optional, Tuple

def process_kinematic_data(df: pd.DataFrame, 
                          required_markers: Optional[Dict[str, Tuple[str, ...]]] = None) -> pd.DataFrame:
    """Process kinematic data with validation."""
    if required_markers is None:
        required_markers = {
            'wrist': ('WRA', 'WRB'),
            'elbow': ('ELB_L', 'ELB_M'),
            'shoulder': ('SA_1', 'SA_2', 'SA_3')
        }
    
    # Validate markers exist
    for joint, markers in required_markers.items():
        missing = [m for m in markers if m not in df.columns.levels[0]]
        if missing:
            raise ValueError(f"Missing required markers for {joint}: {missing}")
    
    return df

def filter_data(data: np.ndarray, cutoff: float = 10.0, fs: float = 100.0, 
                order: int = 4, axis: int = 0) -> np.ndarray:
    """Apply Butterworth filter to data array along specified axis."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Handle NaN values by interpolating
    if np.any(np.isnan(data)):
        mask = np.isnan(data)
        x = np.arange(data.shape[axis])
        for i in range(data.shape[1-axis]):
            if axis == 0:
                y = data[:, i]
                mask_i = mask[:, i]
            else:
                y = data[i, :]
                mask_i = mask[i, :]
            
            if np.any(mask_i):
                valid = ~mask_i
                if np.any(valid):
                    interp = np.interp(x, x[valid], y[valid])
                    if axis == 0:
                        data[:, i] = interp
                    else:
                        data[i, :] = interp
    
    return filtfilt(b, a, data, axis=axis)

def calculate_virtual_joints(marker_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate virtual joint centers from markers."""
    virtual_joints = {}
    
    # Virtual Wrist = midpoint of WRA and WRB
    if 'WRA' in marker_data and 'WRB' in marker_data:
        virtual_joints['V_Wrist'] = (marker_data['WRA'] + marker_data['WRB']) / 2
    
    # Virtual Elbow = midpoint of ELB_L and ELB_M
    if 'ELB_L' in marker_data and 'ELB_M' in marker_data:
        virtual_joints['V_Elbow'] = (marker_data['ELB_L'] + marker_data['ELB_M']) / 2
    
    # Virtual Shoulder = centroid of SA_1, SA_2, SA_3
    shoulder_markers = ['SA_1', 'SA_2', 'SA_3']
    if all(m in marker_data for m in shoulder_markers):
        virtual_joints['V_Shoulder'] = np.mean([marker_data[m] for m in shoulder_markers], axis=0)
    
    if not virtual_joints:
        raise ValueError("Could not compute any virtual joints - check marker data")
        
    return virtual_joints