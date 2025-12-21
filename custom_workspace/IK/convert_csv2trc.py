import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys  # <--- Added

# ========================================
# CONFIGURATION - Edit these paths
# ========================================
# INPUT_CSV = "/home/abdul/Desktop/myosuite/custom_workspace/data/kinematic/Stroke/S1_12_1.csv"
INPUT_CSV = "/home/abdul/Desktop/myosuite/custom_workspace/data/kinematic/Healthy/01_12_1.csv"
OUTPUT_TRC = "/home/abdul/Desktop/myosuite/custom_workspace/IK/output/01_12_1.trc"
DATA_RATE = 200.0
VISUALIZE = False
SAVE_PLOT = None

# <--- ADDED: Override defaults if running from batch script
if len(sys.argv) > 1:
    INPUT_CSV = sys.argv[1]
    OUTPUT_TRC = sys.argv[2]
# ========================================

def process_kinematic_data(filepath):
    """
    Loads and processes the MHH kinematic data from a CSV file into a
    structured DataFrame with a multi-level header.
    """
    try:
        header_df = pd.read_csv(filepath, header=None, nrows=2, sep=',')
        marker_names = header_df.iloc[0].str.strip().ffill()
        axis_names = header_df.iloc[1].str.strip()
        multi_index = pd.MultiIndex.from_arrays([marker_names, axis_names])
        data = pd.read_csv(filepath, header=None, skiprows=2, names=multi_index, sep=',')
        return data
    except FileNotFoundError:
        print(f"Error: The file was not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return None

def filter_data(df, cutoff=2, fs=200, order=4):
    """
    Applies a low-pass Butterworth filter to all marker data. Assumes a 200Hz
    sampling rate for the kinematic data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_df = df.copy()
    for col in df.columns:
        # Interpolate to fill any NaNs before filtering
        series = filtered_df[col].interpolate(method='linear', limit_direction='both')
        filtered_df[col] = filtfilt(b, a, series)
        
    return filtered_df

def visualize_filtering(raw_df, filtered_df, output_path=None):
    """
    Visualizes the effect of Butterworth filtering by comparing raw vs filtered data.
    Shows X, Y, Z trajectories for the three main anatomical markers used for virtual joints.
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Butterworth Filter Effect: Raw vs Filtered Data', fontsize=16, fontweight='bold')
    
    # Markers to visualize (the ones used to create virtual joints)
    markers_to_plot = [
        ('WRA', 'Wrist (WRA)'),
        ('ELB_L', 'Elbow Lateral (ELB_L)'),
        ('SA_1', 'Shoulder (SA_1)')
    ]
    
    colors_raw = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, teal, blue
    colors_filtered = ['#C92A2A', '#087F5B', '#1864AB']  # Darker versions
    
    for row, (marker, label) in enumerate(markers_to_plot):
        for col, axis in enumerate(['X', 'Y', 'Z']):
                ax = axes[row, col]

                # Time vector (assuming 200Hz sampling)
                time = np.arange(len(raw_df)) / 200.0

                # Plot raw data (thin dashed with faint markers so it remains visible)
                ax.plot(time, raw_df[(marker, axis)], 
                        color=colors_raw[row], alpha=0.7, linewidth=0.9,
                        label='Raw', linestyle='--', marker='.', markersize=2)

                # Plot filtered data (thicker solid line)
                ax.plot(time, filtered_df[(marker, axis)], 
                        color=colors_filtered[row], linewidth=1.8,
                        label='Filtered', linestyle='-')

                # Formatting
                ax.set_xlabel('Time (s)', fontsize=9)
                ax.set_ylabel(f'{axis} (mm)', fontsize=9)
                ax.set_title(f'{label} - {axis} axis', fontsize=10)
                ax.grid(alpha=0.3, linestyle='--')
                ax.legend(loc='upper right', fontsize=8)

                # Add stats text box
                raw_std = raw_df[(marker, axis)].std()
                filtered_std = filtered_df[(marker, axis)].std()
                if raw_std == 0 or np.isnan(raw_std):
                    stats_text = 'Noise ↓: N/A'
                else:
                    noise_reduction = ((raw_std - filtered_std) / raw_std) * 100
                    stats_text = f'Noise ↓: {noise_reduction:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Filter comparison plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_virtual_joints(df):
    """
    Calculates the virtual joint centers (Shoulder, Elbow, Wrist) from the
    anatomical markers to create a more accurate biomechanical model.
    """
    processed_df = df.copy()
    
    # Virtual Wrist Center: Midpoint between WRA (thumb-side) and WRB (pinky-side)
    processed_df[('V_Wrist', 'X')] = (processed_df['WRA']['X'] + processed_df['WRB']['X']) / 2
    processed_df[('V_Wrist', 'Y')] = (processed_df['WRA']['Y'] + processed_df['WRB']['Y']) / 2
    processed_df[('V_Wrist', 'Z')] = (processed_df['WRA']['Z'] + processed_df['WRB']['Z']) / 2
    
    # Virtual Elbow Center: Midpoint between ELB_L (lateral) and ELB_M (medial)
    processed_df[('V_Elbow', 'X')] = (processed_df['ELB_L']['X'] + processed_df['ELB_M']['X']) / 2
    processed_df[('V_Elbow', 'Y')] = (processed_df['ELB_L']['Y'] + processed_df['ELB_M']['Y']) / 2
    processed_df[('V_Elbow', 'Z')] = (processed_df['ELB_L']['Z'] + processed_df['ELB_M']['Z']) / 2

    # Virtual Shoulder Center: Centroid of the three acromial markers
    processed_df[('V_Shoulder', 'X')] = (processed_df['SA_1']['X'] + processed_df['SA_2']['X'] + processed_df['SA_3']['X']) / 3
    processed_df[('V_Shoulder', 'Y')] = (processed_df['SA_1']['Y'] + processed_df['SA_2']['Y'] + processed_df['SA_3']['Y']) / 3
    processed_df[('V_Shoulder', 'Z')] = (processed_df['SA_1']['Z'] + processed_df['SA_2']['Z'] + processed_df['SA_3']['Z']) / 3
    
    return processed_df

def save_to_trc(df, trc_filepath, data_rate=200):
    """
    Saves the processed DataFrame to a .trc (Track Row Column) file format
    that is compatible with MyoSuite and OpenSim.
    """
    markers_to_save = ['V_Shoulder', 'V_Elbow', 'V_Wrist']
    
    trc_df = df[markers_to_save]
    
    num_markers = len(markers_to_save)
    num_frames = len(trc_df)
    
    data_out = pd.DataFrame()
    data_out['Frame#'] = range(1, num_frames + 1)
    data_out['Time'] = trc_df.index / data_rate
    
    for marker in markers_to_save:
        for coord in ['X', 'Y', 'Z']:
            data_out[f'{marker}_{coord}'] = trc_df[(marker, coord)].values

    # Write the TRC file header
    with open(trc_filepath, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_filepath}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{data_rate}\t{data_rate}\t{num_frames}\t{num_markers}\tmm\t{data_rate}\t1\t{num_frames}\n")
        f.write("Frame#\tTime\t" + "\t".join([f"{marker}\t\t" for marker in markers_to_save]) + "\n")
        f.write("\t\t" + "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)]) + "\n")
        f.write("\n")

    data_out.to_csv(trc_filepath, sep='\t', index=False, header=False, mode='a', lineterminator='\n')
    print(f"✓ Successfully saved refined data to: {trc_filepath}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_TRC) if os.path.dirname(OUTPUT_TRC) else '.', exist_ok=True)

    # Load and process the raw data
    raw_kinematic_df = process_kinematic_data(INPUT_CSV)

    if raw_kinematic_df is not None:
        print("1. Successfully loaded raw kinematic data.")
        
        # Filter the data to remove noise
        filtered_df = filter_data(raw_kinematic_df)
        print("2. Applied Butterworth filter to smooth the data.")
        
        # Visualize filtering effect if requested
        if VISUALIZE or SAVE_PLOT:
            print("Generating filtering comparison visualization...")
            visualize_filtering(raw_kinematic_df, filtered_df, 
                              output_path=SAVE_PLOT if SAVE_PLOT else None)
        
        # Calculate virtual joint centers for accuracy
        refined_df = calculate_virtual_joints(filtered_df)
        print("3. Calculated virtual joint centers (V_Shoulder, V_Elbow, V_Wrist).")
        
        # Save the final data to a .trc file
        save_to_trc(refined_df, OUTPUT_TRC, data_rate=DATA_RATE)
        
        print("\nPipeline complete. You are now ready to proceed with Inverse Kinematics.")
