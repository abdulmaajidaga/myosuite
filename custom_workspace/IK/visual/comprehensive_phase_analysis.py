import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from pandas.plotting import parallel_coordinates
import os
import glob
import json
import math

# --- Data Processing Functions ---

def load_and_process_data(filepath):
    try:
        header_df = pd.read_csv(filepath, header=None, nrows=2, sep=',')
        marker_names = header_df.iloc[0].str.strip().ffill()
        axis_names = header_df.iloc[1].str.strip()
        multi_index = pd.MultiIndex.from_arrays([marker_names, axis_names])
        data = pd.read_csv(filepath, header=None, skiprows=2, names=multi_index, sep=',')
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.interpolate(method='linear', limit_direction='both').fillna(0)
        return data
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def calculate_virtual_joints(df):
    processed_df = df.copy()
    try:
        # Wrist (Midpoint of WRA, WRB)
        processed_df[('V_Wrist', 'X')] = (processed_df['WRA']['X'] + processed_df['WRB']['X']) / 2
        processed_df[('V_Wrist', 'Y')] = (processed_df['WRA']['Y'] + processed_df['WRB']['Y']) / 2
        processed_df[('V_Wrist', 'Z')] = (processed_df['WRA']['Z'] + processed_df['WRB']['Z']) / 2
        
        # Elbow (Midpoint of ELB_L, ELB_M)
        processed_df[('V_Elbow', 'X')] = (processed_df['ELB_L']['X'] + processed_df['ELB_M']['X']) / 2
        processed_df[('V_Elbow', 'Y')] = (processed_df['ELB_L']['Y'] + processed_df['ELB_M']['Y']) / 2
        processed_df[('V_Elbow', 'Z')] = (processed_df['ELB_L']['Z'] + processed_df['ELB_M']['Z']) / 2

        # Shoulder (Centroid of SA_1, SA_2, SA_3)
        processed_df[('V_Shoulder', 'X')] = (processed_df['SA_1']['X'] + processed_df['SA_2']['X'] + processed_df['SA_3']['X']) / 3
        processed_df[('V_Shoulder', 'Y')] = (processed_df['SA_1']['Y'] + processed_df['SA_2']['Y'] + processed_df['SA_3']['Y']) / 3
        processed_df[('V_Shoulder', 'Z')] = (processed_df['SA_1']['Z'] + processed_df['SA_2']['Z'] + processed_df['SA_3']['Z']) / 3
        
        return processed_df
    except KeyError:
        return None

def calculate_velocity(positions, fs=200.0):
    d_pos = np.diff(positions, axis=0)
    dist = np.linalg.norm(d_pos, axis=1)
    velocity = dist * fs
    return np.insert(velocity, 0, 0)

def extract_features(csv_files, phase_data, fs):
    """
    Extracts a feature DataFrame for Heatmap/Parallel Coordinates.
    """
    feature_rows = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) != 3: continue
        
        df = load_and_process_data(filepath)
        if df is None: continue
        joints = calculate_virtual_joints(df)
        if joints is None: continue
        
        wrist_pos = joints['V_Wrist'].values
        velocity = calculate_velocity(wrist_pos, fs)
        
        # Phase Ranges
        # Reach, Lift, Place, Rest
        ranges = {
            'Reach': (0, idxs[0]), 
            'Lift': (idxs[0], idxs[1]), 
            'Place': (idxs[1], idxs[2]),
            'Rest': (idxs[2], len(wrist_pos))
        }
        
        row = {'Filename': filename}
        for phase, (start, end) in ranges.items():
            if start >= end: 
                # Handle empty phases gracefully
                row[f'{phase}_Duration'] = 0
                row[f'{phase}_PeakVel'] = 0
                row[f'{phase}_PathLen'] = 0
                continue
            
            # Duration
            row[f'{phase}_Duration'] = (end - start) / fs
            
            # Peak Velocity
            vel_seg = velocity[start:end]
            row[f'{phase}_PeakVel'] = np.max(vel_seg) if len(vel_seg) > 0 else 0
            
            # Path Length
            pos_seg = wrist_pos[start:end]
            if len(pos_seg) > 1:
                path_len = np.sum(np.linalg.norm(np.diff(pos_seg, axis=0), axis=1))
                row[f'{phase}_PathLen'] = path_len
            else:
                row[f'{phase}_PathLen'] = 0
                
        feature_rows.append(row)
        
    return pd.DataFrame(feature_rows)

# --- Visualization Functions ---

def generate_heatmap(features_df, output_path):
    print(f"Generating Heatmap...")
    
    # Normalize features for display (0-1 scaling per column)
    df_numeric = features_df.drop(columns=['Filename'])
    # Handle columns with 0 variance (e.g. if Rest is always 0)
    df_norm = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    df_norm = df_norm.fillna(0) # In case max == min
    
    plt.figure(figsize=(14, 10))
    plt.imshow(df_norm.values, aspect='auto', cmap='coolwarm', interpolation='nearest')
    
    # Axis labels
    plt.xticks(range(len(df_norm.columns)), df_norm.columns, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(features_df)), features_df['Filename'], fontsize=8)
    
    plt.colorbar(label='Normalized Value (Blue=Low, Red=High)')
    plt.title("Feature Heatmap (Normalized) including Rest", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Heatmap saved to {output_path}")

def generate_parallel_coordinates(features_df, output_path):
    print(f"Generating Parallel Coordinates Plot...")
    
    # Normalize for plotting so axes are comparable
    df_norm = features_df.copy()
    cols = df_norm.columns.drop('Filename')
    for col in cols:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 0
    
    plt.figure(figsize=(16, 8))
    pd.plotting.parallel_coordinates(df_norm, 'Filename', color='teal', alpha=0.3, linewidth=1.5)
    plt.legend().remove() # Too many filenames to list in legend
    
    plt.xticks(rotation=45, ha='right')
    plt.title("Parallel Coordinates (All Files normalized)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Parallel Coordinates saved to {output_path}")

def generate_time_normalized_plot(csv_files, phase_data, fs, output_path):
    print(f"Generating Time-Normalized Kinematics Plot...")
    
    # Included Rest phase as well
    phases = ['Reach', 'Lift', 'Place', 'Rest']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
    
    for i, phase_name in enumerate(phases):
        ax = axes[i]
        all_profiles = []
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            idxs = phase_data.get(filename, [])
            if len(idxs) != 3: continue
            
            df = load_and_process_data(filepath)
            if df is None: continue
            joints = calculate_virtual_joints(df)
            wrist_pos = joints['V_Wrist'].values
            velocity = calculate_velocity(wrist_pos, fs)
            
            # Select range
            if i == 0: start, end = 0, idxs[0]
            elif i == 1: start, end = idxs[0], idxs[1]
            elif i == 2: start, end = idxs[1], idxs[2]
            else: start, end = idxs[2], len(velocity) # Rest
            
            if start >= end: continue
            
            vel_seg = velocity[start:end]
            if len(vel_seg) < 2: continue
            
            # Normalize Time to 100 points
            x_old = np.linspace(0, 1, len(vel_seg))
            x_new = np.linspace(0, 1, 100)
            vel_norm = np.interp(x_new, x_old, vel_seg)
            
            all_profiles.append(vel_norm)
            ax.plot(x_new * 100, vel_norm, color='gray', alpha=0.2)
            
        # Plot Mean Line
        if all_profiles:
            mean_profile = np.mean(all_profiles, axis=0)
            std_profile = np.std(all_profiles, axis=0)
            
            ax.plot(x_new * 100, mean_profile, color='red', linewidth=2.5, label='Mean')
            ax.fill_between(x_new * 100, mean_profile - std_profile, mean_profile + std_profile, 
                            color='red', alpha=0.1)
            
        ax.set_title(f"{phase_name} Phase Velocity")
        ax.set_xlabel("% Phase Completion")
        if i == 0: ax.set_ylabel("Velocity (mm/s)")
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Time-Normalized Velocity Profiles (Kinematic Consistency)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Time-Normalized Plot saved to {output_path}")


# --- Existing Functions (Reused) ---
# (generate_2d_trajectories, generate_3d_poses, generate_phase_metrics, generate_3d_phase_paths)
# I will include them here to keep the file complete.

def generate_2d_trajectories(csv_files, phase_data, fs, output_path):
    print(f"Generating 2D Position Analysis...")
    num_files = len(csv_files)
    cols = 5; rows = math.ceil(num_files / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 18), constrained_layout=True)
    axes = axes.flatten()
    colors = {'X': 'red', 'Y': 'green', 'Z': 'blue', 'Phase1': '#e6ffe6', 'Phase2': '#fff0e6', 'Phase3': '#e6e6ff', 'Phase4': '#f2f2f2'}
    
    for i, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        ax = axes[i]
        df = load_and_process_data(filepath)
        if df is not None:
            joints = calculate_virtual_joints(df)
            if joints is not None:
                wx = joints['V_Wrist']['X']; wy = joints['V_Wrist']['Y']; wz = joints['V_Wrist']['Z']
                time = np.arange(len(wx)) / fs
                ax.plot(time, wx, color=colors['X'], lw=1, label='X')
                ax.plot(time, wy, color=colors['Y'], lw=1, label='Y')
                ax.plot(time, wz, color=colors['Z'], lw=1, label='Z')
                idxs = phase_data.get(filename, [])
                if len(idxs) == 3:
                    t_phases = [idx/fs for idx in idxs]; t_end = time[-1]
                    ax.axvspan(0, t_phases[0], color=colors['Phase1'], alpha=0.5)
                    ax.axvspan(t_phases[0], t_phases[1], color=colors['Phase2'], alpha=0.5)
                    ax.axvspan(t_phases[1], t_phases[2], color=colors['Phase3'], alpha=0.5)
                    ax.axvspan(t_phases[2], t_end, color=colors['Phase4'], alpha=0.5)
        ax.set_title(filename, fontsize=9)
        if i == 0:
             # Explicit legend for X, Y, Z and Phases
             handles = [
                plt.Line2D([0], [0], color=colors['X'], lw=2, label='X (mm)'),
                plt.Line2D([0], [0], color=colors['Y'], lw=2, label='Y (mm)'),
                plt.Line2D([0], [0], color=colors['Z'], lw=2, label='Z (mm)'),
                mpatches.Patch(color=colors['Phase1'], label='Reach'),
                mpatches.Patch(color=colors['Phase2'], label='Lift'),
                mpatches.Patch(color=colors['Phase3'], label='Place'),
                mpatches.Patch(color=colors['Phase4'], label='Rest')
             ]
             ax.legend(handles=handles, loc='upper right', fontsize=8)

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    
    # Global Labels
    fig.text(0.5, 0.01, 'Time (seconds)', ha='center', fontsize=14)
    fig.text(0.01, 0.5, 'Position (mm)', va='center', rotation='vertical', fontsize=14)
    
    plt.suptitle("Wrist Position Trajectories (X, Y, Z) by Phase", fontsize=16)
    plt.savefig(output_path, dpi=150)
    print(f"2D Analysis saved to {output_path}")

def generate_3d_poses(csv_files, phase_data, output_path):
    print(f"Generating 3D Pose Aggregation...")
    # poses[0] = Start/Rest, [1] = Pick, [2] = Mouth, [3] = Place
    poses = {0: [], 1: [], 2: [], 3: []}
    
    for filepath in csv_files:
        filename = os.path.basename(filepath); idxs = phase_data.get(filename, [])
        if len(idxs) != 3: continue
        df = load_and_process_data(filepath); joints = calculate_virtual_joints(df) if df is not None else None
        if joints is not None:
            # We need 4 instants: Start(0), Pick(idxs[0]), Mouth(idxs[1]), Place(idxs[2])
            events = [0, idxs[0], idxs[1], idxs[2]]
            
            for i, idx in enumerate(events):
                try:
                    s = joints['V_Shoulder'].iloc[idx].values
                    e = joints['V_Elbow'].iloc[idx].values
                    w = joints['V_Wrist'].iloc[idx].values
                    # Normalize: Pin Shoulder to (0,0,0)
                    poses[i].append((s-s, e-s, w-s))
                except IndexError: pass
    
    fig = plt.figure(figsize=(24, 6))
    titles = ["Start / Rest Position", "Event: Bottle Pick", "Event: At Mouth", "Event: Place Back"]
    colors = ['gray', 'green', 'orange', 'blue']
    
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        for (s, e, w) in poses[i]:
            ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color='gray', alpha=0.2)
            ax.plot([e[0],w[0]], [e[1],w[1]], [e[2],w[2]], color=colors[i], alpha=0.5, linewidth=2)
            ax.scatter(*s, color='black', s=20) # Shoulder
            ax.scatter(*e, color='gray', s=10)  # Elbow
            ax.scatter(*w, color=colors[i], s=30) # Wrist
            
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xlim([-600,600]); ax.set_ylim([-600,600]); ax.set_zlim([-600,600])
        ax.view_init(elev=20, azim=45)
        
    plt.suptitle("Aggregated Arm Postures at Key Events (Aligned to Shoulder Origin)\nThis allows comparing arm geometry (extension/flexion) independent of subject position.", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"3D Analysis saved to {output_path}")

def generate_phase_metrics_boxplot(features_df, output_path):
    # Re-implementing boxplots using the DataFrame for consistency
    print(f"Generating Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['Duration', 'PeakVel', 'PathLen']
    titles = ['Duration (s)', 'Peak Velocity (mm/s)', 'Path Length (mm)']
    phases = ['Reach', 'Lift', 'Place', 'Rest']
    colors = ['lightgreen', 'moccasin', 'lightblue', 'lightgray']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Check if column exists (Rest columns might be missing if I didn't re-run extract_features yet?)
        # They will be present because extract_features was updated.
        data = []
        labels = []
        for p in phases:
            col_name = f'{p}_{metric}'
            if col_name in features_df.columns:
                data.append(features_df[col_name].dropna().values)
                labels.append(p)
        
        if not data: continue

        bplot = ax.boxplot(data, patch_artist=True, labels=labels)
        for patch, color in zip(bplot['boxes'], colors): patch.set_facecolor(color)
        ax.set_title(titles[i], fontweight='bold'); ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(output_path, dpi=150)

def generate_3d_phase_paths(csv_files, phase_data, output_path):
    print(f"Generating 3D Phase Paths...")
    fig = plt.figure(figsize=(10, 10)); ax = fig.add_subplot(111, projection='3d')
    phase_colors = ['green', 'orange', 'blue']; ranges_idx = [(0,0), (0,1), (1,2)] # Needs fixing logic below
    
    for filepath in csv_files:
        filename = os.path.basename(filepath); idxs = phase_data.get(filename, [])
        if len(idxs) != 3: continue
        df = load_and_process_data(filepath); joints = calculate_virtual_joints(df) if df is not None else None
        if joints is not None:
            s_pos = joints['V_Shoulder'].values; w_pos = joints['V_Wrist'].values; rel_w = w_pos - s_pos
            ranges = [(0, idxs[0]), (idxs[0], idxs[1]), (idxs[1], idxs[2])]
            for i, (start, end) in enumerate(ranges):
                if len(rel_w[start:end]) > 1:
                    ax.plot(rel_w[start:end,0], rel_w[start:end,1], rel_w[start:end,2], color=phase_colors[i], alpha=0.2)
    ax.set_xlim([-600,600]); ax.set_ylim([-600,600]); ax.set_zlim([-600,600]); ax.view_init(elev=20, azim=45)
    plt.savefig(output_path, dpi=150)


# --- Main ---

import argparse

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    # Parse Args
    parser = argparse.ArgumentParser(description="Comprehensive Kinematic Phase Analysis.")
    parser.add_argument("--dataset", type=str, choices=['healthy', 'stroke'], default='stroke', help="Dataset to process.")
    args = parser.parse_args()
    
    # Determine Paths
    if args.dataset == 'healthy':
        filtered_dir = os.path.join(project_root, "data", "kinematic", "Healthy", "filtered")
        output_dir = os.path.join(script_dir, "healthy")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_path = os.path.join(output_dir, "manual_phase_indices.json")
    else: # stroke
        filtered_dir = os.path.join(project_root, "data", "kinematic", "Stroke", "filtered")
        output_dir = script_dir
        json_path = os.path.join(script_dir, "manual_phase_indices.json")
    
    # Outputs
    out_2d = os.path.join(output_dir, "analysis_1_wrist_trajectories.png")
    out_3d_poses = os.path.join(output_dir, "analysis_2_poses_3d.png")
    out_boxplot = os.path.join(output_dir, "analysis_3_boxplots.png")
    out_3d_paths = os.path.join(output_dir, "analysis_4_paths_3d.png")
    out_heatmap = os.path.join(output_dir, "analysis_5_heatmap.png")
    out_norm_time = os.path.join(output_dir, "analysis_6_norm_time.png")
    out_parallel = os.path.join(output_dir, "analysis_7_parallel_coords.png")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run interactive_phase_selector.py --dataset {args.dataset} first.")
        return
        
    with open(json_path, 'r') as f:
        phase_data = json.load(f)
        
    csv_files = sorted(glob.glob(os.path.join(filtered_dir, "*.csv")))[:25]
    if not csv_files:
        print(f"No CSV files found in {filtered_dir}")
        return

    print(f"Running Analysis on {args.dataset.upper()} dataset...")
    
    # 1. Extract Features DataFrame
    features_df = extract_features(csv_files, phase_data, 200.0)
    
    # 2. Generate Graphs
    generate_heatmap(features_df, out_heatmap)
    generate_time_normalized_plot(csv_files, phase_data, 200.0, out_norm_time)
    generate_phase_metrics_boxplot(features_df, out_boxplot)
    generate_3d_phase_paths(csv_files, phase_data, out_3d_paths)
    generate_parallel_coordinates(features_df, out_parallel)
    
    # Existing ones
    generate_2d_trajectories(csv_files, phase_data, 200.0, out_2d)
    generate_3d_poses(csv_files, phase_data, out_3d_poses)
    
    print(f"\nAll 7 Analysis Charts Generated Successfully in: {output_dir}")

if __name__ == "__main__":
    main()
