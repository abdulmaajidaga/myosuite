"""
3D marker animation of OpenSim TRC files.

What it does:
  Parses a .trc file (tab-separated, 5-row header, marker triplets X/Y/Z),
  extracts named marker positions, and renders an animated 3D scatter plot
  showing all markers moving through the recording.

Input:
  - output/originals/trc/*.trc or output/generated/trc/*.trc
    (OpenSim TRC format, 200 Hz, mm)

Output:
  - Interactive matplotlib 3D animation window (no file saved)

Usage:
  python scripts/viz/render/visualise_trc.py
  (defaults to first .trc found in output/originals/trc/)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob

def parse_trc(filepath):
    """
    Parses an OpenSim .trc file into a pandas DataFrame.
    """
    print(f"Loading: {os.path.basename(filepath)}...")
    try:
        # TRC format usually has 3 header rows, then specific columns
        # Row 1: Header info (PathFileType...)
        # Row 2: Header info (DataRate...)
        # Row 3: Metadata values (200.0...)
        # Row 4: Marker Names (Frame# Time V_Shoulder...)
        # Row 5: Coordinates (X1 Y1 Z1...)
        
        # We skip to Row 4 (index 3) to get marker names
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Extract Marker Names from line 3 (tab separated)
        # Filter out empty strings and 'Frame#', 'Time'
        raw_header = lines[3].strip().split('\t')
        marker_names = [m for m in raw_header if m and m not in ['Frame#', 'Time']]
        
        # Read the data block starting from line 6 (index 5)
        # The columns are: Frame, Time, M1_X, M1_Y, M1_Z, M2_X...
        data = pd.read_csv(filepath, sep='\t', skiprows=5, header=None)
        
        # Structure data into a dictionary of {MarkerName: (N, 3) array}
        markers = {}
        for i, name in enumerate(marker_names):
            # Columns 0=Frame, 1=Time.
            # Marker 1 starts at col 2, 3, 4. Marker 2 at 5, 6, 7.
            start_col = 2 + (i * 3)
            # Extract X, Y, Z
            xyz = data.iloc[:, start_col:start_col+3].values
            markers[name] = xyz
            
        return markers, data.iloc[:, 1].values # Markers dict, Time array
        
    except Exception as e:
        print(f"Error parsing TRC: {e}")
        return None, None

def animate_all_trcs(file_list):
    print(f"Found {len(file_list)} TRC files. Loading...")
    
    all_data = []
    max_frames = 0
    
    for fpath in file_list:
        markers, time = parse_trc(fpath)
        if markers:
            frames = len(time)
            if frames > max_frames: max_frames = frames
            all_data.append({
                'name': os.path.basename(fpath),
                'markers': markers,
                'frames': frames,
                'time': time
            })
            
    if not all_data:
        print("No valid data loaded.")
        return

    # Setup Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate limits based on a sample of points to ensure everything is visible
    sample_points = []
    for d in all_data:
        # Take every 10th frame of Shoulder to estimate bounds
        if 'V_Shoulder' in d['markers']:
            sample_points.append(d['markers']['V_Shoulder'][::10])
            
    if sample_points:
        stacked = np.vstack(sample_points)
        mid = np.mean(stacked, axis=0)
        rng = np.max(np.ptp(stacked, axis=0)) / 2.0
        # Ensure minimum range
        if rng < 100: rng = 500 
        
        ax.set_xlim(mid[0] - rng, mid[0] + rng)
        ax.set_ylim(mid[1] - rng, mid[1] + rng)
        ax.set_zlim(mid[2] - rng, mid[2] + rng)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"Visualizing {len(all_data)} TRC Files")

    # Create lines
    lines = []
    colors = plt.cm.jet(np.linspace(0, 1, len(all_data)))
    
    for i, d in enumerate(all_data):
        # Arm
        ln, = ax.plot([], [], [], 'o-', lw=2, color=colors[i], alpha=0.6, label=d['name'])
        # Hand
        ln_h, = ax.plot([], [], [], '-', lw=1, color=colors[i], alpha=0.6)
        lines.append((ln, ln_h))
        
    # Legend (only if few)
    if len(all_data) <= 15:
        ax.legend(fontsize='x-small')

    # Time Display
    time_text = ax.text2D(0.05, 0.95, "Time: 0.00 s", transform=ax.transAxes, fontsize=12)

    def update(frame):
        # Update Time
        t_val = 0.0
        for d in all_data:
            if frame < d['frames']:
                t_val = d['time'][frame]
                break
        time_text.set_text(f"Time: {t_val:.3f} s")

        for i, d in enumerate(all_data):
            # Loop animation
            idx = frame % d['frames']
            
            markers = d['markers']
            ln, ln_h = lines[i]
            
            if 'V_Shoulder' in markers and 'V_Elbow' in markers and 'V_Wrist' in markers:
                sh = markers['V_Shoulder'][idx]
                el = markers['V_Elbow'][idx]
                wr = markers['V_Wrist'][idx]
                
                ln.set_data([sh[0], el[0], wr[0]], [sh[1], el[1], wr[1]])
                ln.set_3d_properties([sh[2], el[2], wr[2]])
                
                # Hand
                pk = None
                if 'V_Vector' in markers: pk = markers['V_Vector'][idx]
                elif 'WRB' in markers: pk = markers['WRB'][idx]
                
                if pk is not None:
                    ln_h.set_data([wr[0], pk[0]], [wr[1], pk[1]])
                    ln_h.set_3d_properties([wr[2], pk[2]])
                    
        return [l for pair in lines for l in pair]

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=False)
    plt.show()

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    from src.utils.config import get_path
    trc_dir = get_path("output_originals_trc")
    # Search in TRC output directory
    files = glob.glob(os.path.join(trc_dir, "*.trc"))
    files += glob.glob(os.path.join(trc_dir, "**", "*.trc"), recursive=True)

    files = sorted(list(set(files)))

    if not files:
        print(f"No .trc files found in {trc_dir}")
    else:
        animate_all_trcs(files)