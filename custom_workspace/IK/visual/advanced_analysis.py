"""
Advanced Kinematic Analysis
Generates 10 analysis outputs from raw motion capture data.
"""
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
import argparse
from scipy.signal import butter, filtfilt

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
        processed_df[('V_Wrist', 'X')] = (processed_df['WRA']['X'] + processed_df['WRB']['X']) / 2
        processed_df[('V_Wrist', 'Y')] = (processed_df['WRA']['Y'] + processed_df['WRB']['Y']) / 2
        processed_df[('V_Wrist', 'Z')] = (processed_df['WRA']['Z'] + processed_df['WRB']['Z']) / 2
        processed_df[('V_Elbow', 'X')] = (processed_df['ELB_L']['X'] + processed_df['ELB_M']['X']) / 2
        processed_df[('V_Elbow', 'Y')] = (processed_df['ELB_L']['Y'] + processed_df['ELB_M']['Y']) / 2
        processed_df[('V_Elbow', 'Z')] = (processed_df['ELB_L']['Z'] + processed_df['ELB_M']['Z']) / 2
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

# --- Advanced Kinematic Calculations ---

def calculate_angles(joints_df):
    """Calculates Elbow Flexion and Shoulder Elevation angles."""
    s = joints_df['V_Shoulder'].values
    e = joints_df['V_Elbow'].values
    w = joints_df['V_Wrist'].values

    v_upper = e - s
    v_fore = w - e
    v_vert = np.array([0, 0, 1])

    angles = {}

    # Elbow Flexion
    norms_upper = np.linalg.norm(v_upper, axis=1)
    norms_fore = np.linalg.norm(v_fore, axis=1)
    dot_prod = np.sum(v_upper * v_fore, axis=1)
    cosine_angle = np.clip(dot_prod / (norms_upper * norms_fore), -1.0, 1.0)
    angles['Elbow_Flexion'] = np.degrees(np.arccos(cosine_angle))

    # Shoulder Elevation
    dot_prod_s = np.sum(v_upper * v_vert, axis=1)
    cos_s = np.clip(dot_prod_s / norms_upper, -1.0, 1.0)
    angles['Shoulder_Elevation'] = 180 - np.degrees(np.arccos(cos_s))

    return pd.DataFrame(angles)

def calculate_derivatives(pos, fs):
    """Calc Vel, Acc, Jerk magnitude"""
    vel = np.diff(pos, axis=0) * fs
    acc = np.diff(vel, axis=0) * fs
    jerk = np.diff(acc, axis=0) * fs

    vel = np.vstack([np.zeros((1,3)), vel])
    acc = np.vstack([np.zeros((2,3)), acc])
    jerk = np.vstack([np.zeros((3,3)), jerk])

    return vel, acc, jerk

def calculate_smoothness(jerk, duration, v_peak):
    """Log Dimensionless Jerk"""
    if duration <= 0 or v_peak <= 0: return 0
    jerk_sq = np.sum(jerk**2, axis=1)
    integral_jerk = np.sum(jerk_sq) * (1/200.0)
    dj = (integral_jerk * (duration**3)) / (v_peak**2)
    if dj <= 0: return 0
    return -np.log(dj)

def calculate_straightness(pos):
    """Index of Curvature: Path Length / Straight Line Dist"""
    if len(pos) < 2: return 1.0
    path_len = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    straight_dist = np.linalg.norm(pos[-1] - pos[0])
    if straight_dist == 0: return 1.0
    return path_len / straight_dist

# --- Feature Extraction ---

def extract_features(csv_files, phase_data, fs):
    """Extracts a feature DataFrame for Heatmap/Parallel Coordinates."""
    feature_rows = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) < 3: continue

        df = load_and_process_data(filepath)
        if df is None: continue
        joints = calculate_virtual_joints(df)
        if joints is None: continue

        wrist_pos = joints['V_Wrist'].values
        velocity = calculate_velocity(wrist_pos, fs)
        n_frames = len(wrist_pos)

        # Phase Ranges (3-point: Pick, Drink, Place)
        ranges = {
            'Pick': (0, idxs[0]),
            'Drink': (idxs[0], idxs[1]),
            'Place': (idxs[1], idxs[2])
        }

        row = {'Filename': filename}
        for phase, (start, end) in ranges.items():
            if start >= end:
                row[f'{phase}_Duration'] = 0
                row[f'{phase}_PeakVel'] = 0
                row[f'{phase}_PathLen'] = 0
                continue

            row[f'{phase}_Duration'] = (end - start) / fs
            vel_seg = velocity[start:end]
            row[f'{phase}_PeakVel'] = np.max(vel_seg) if len(vel_seg) > 0 else 0
            pos_seg = wrist_pos[start:end]
            if len(pos_seg) > 1:
                row[f'{phase}_PathLen'] = np.sum(np.linalg.norm(np.diff(pos_seg, axis=0), axis=1))
            else:
                row[f'{phase}_PathLen'] = 0

        feature_rows.append(row)

    return pd.DataFrame(feature_rows)

def extract_advanced_features(csv_files, phase_data, fs):
    """Extracts advanced kinematic features."""
    features = []
    cyclograms = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) < 3: continue

        df = load_and_process_data(filepath)
        if df is None: continue
        joints = calculate_virtual_joints(df)
        if joints is None: continue

        angles = calculate_angles(joints)
        w_pos = joints['V_Wrist'].values
        n_frames = len(w_pos)
        vel, acc, jerk = calculate_derivatives(w_pos, fs)
        vel_mag = np.linalg.norm(vel, axis=1)

        # Pick phase: 0 to idxs[0]
        p_start, p_end = 0, idxs[0]
        pick_angles = angles.iloc[p_start:p_end]

        elbow_rom = pick_angles['Elbow_Flexion'].max() - pick_angles['Elbow_Flexion'].min() if not pick_angles.empty else 0

        s_pos = joints['V_Shoulder'].values
        s_disp = np.linalg.norm(s_pos - s_pos[0], axis=1)
        max_trunk_comp = np.max(s_disp)

        cyclograms.append((angles['Shoulder_Elevation'].values, angles['Elbow_Flexion'].values, filename))

        pick_w_pos = w_pos[p_start:p_end]
        if len(pick_w_pos) > 1:
            wrist_dist = np.sum(np.linalg.norm(np.diff(pick_w_pos, axis=0), axis=1))
            d_shoulder = np.sum(np.abs(np.diff(pick_angles['Shoulder_Elevation'])))
            d_elbow = np.sum(np.abs(np.diff(pick_angles['Elbow_Flexion'])))
            joint_travel = d_shoulder + d_elbow
            efficiency = wrist_dist / joint_travel if joint_travel > 0 else 0
            straightness = calculate_straightness(pick_w_pos)
        else:
            efficiency = 0
            straightness = 1.0

        if len(vel_mag[p_start:p_end]) > 0:
            peak_idx = np.argmax(vel_mag[p_start:p_end])
            time_to_peak = peak_idx / len(vel_mag[p_start:p_end])
        else:
            time_to_peak = 0

        # Drink phase: idxs[0] to idxs[1]
        d_start, d_end = idxs[0], idxs[1]
        if d_end > d_start:
            drink_dur = (d_end - d_start) / fs
            drink_vpeak = np.max(vel_mag[d_start:d_end])
            drink_jerk = jerk[d_start:d_end]
            smoothness = calculate_smoothness(drink_jerk, drink_dur, drink_vpeak)
            drink_angles = angles.iloc[d_start:d_end]
            max_sh_elev = drink_angles['Shoulder_Elevation'].max() if len(drink_angles) > 0 else 0
        else:
            smoothness = 0
            max_sh_elev = 0

        # Place phase end
        place_end = idxs[2]
        final_pos = w_pos[place_end] if place_end < len(w_pos) else w_pos[-1]

        features.append({
            'Filename': filename,
            'Elbow_ROM_Pick': elbow_rom,
            'Trunk_Compensation': max_trunk_comp,
            'Efficiency_Ratio': efficiency,
            'Straightness_Index': straightness,
            'Time_to_Peak_Vel': time_to_peak,
            'Drink_Smoothness_LDLJ': smoothness,
            'Max_Shoulder_Elev_Drink': max_sh_elev,
            'Final_X': final_pos[0], 'Final_Y': final_pos[1], 'Final_Z': final_pos[2]
        })

    return pd.DataFrame(features), cyclograms

# --- Visualization Functions ---

def generate_heatmap(features_df, output_path):
    print("  1/10: Heatmap...")
    df_numeric = features_df.drop(columns=['Filename'])
    df_norm = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())
    df_norm = df_norm.fillna(0)

    plt.figure(figsize=(14, 10))
    plt.imshow(df_norm.values, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.xticks(range(len(df_norm.columns)), df_norm.columns, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(features_df)), features_df['Filename'], fontsize=8)
    plt.colorbar(label='Normalized Value')
    plt.title("Feature Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_parallel_coordinates(features_df, output_path):
    print("  2/10: Parallel Coordinates...")
    df_norm = features_df.copy()
    cols = df_norm.columns.drop('Filename')
    for col in cols:
        min_val, max_val = df_norm[col].min(), df_norm[col].max()
        df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val) if max_val > min_val else 0

    plt.figure(figsize=(16, 8))
    pd.plotting.parallel_coordinates(df_norm, 'Filename', color='teal', alpha=0.3, linewidth=1.5)
    plt.legend().remove()
    plt.xticks(rotation=45, ha='right')
    plt.title("Parallel Coordinates", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_time_normalized_plot(csv_files, phase_data, fs, output_path):
    print("  3/10: Time-Normalized Velocity...")
    phases = ['Pick', 'Drink', 'Place']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, phase_name in enumerate(phases):
        ax = axes[i]
        all_profiles = []

        for filepath in csv_files:
            filename = os.path.basename(filepath)
            idxs = phase_data.get(filename, [])
            if len(idxs) < 3: continue

            df = load_and_process_data(filepath)
            if df is None: continue
            joints = calculate_virtual_joints(df)
            if joints is None: continue
            wrist_pos = joints['V_Wrist'].values
            velocity = calculate_velocity(wrist_pos, fs)

            # 3-point indices: Pick, Drink, Place
            if i == 0: start, end = 0, idxs[0]
            elif i == 1: start, end = idxs[0], idxs[1]
            else: start, end = idxs[1], idxs[2]

            if start >= end: continue
            vel_seg = velocity[start:end]
            if len(vel_seg) < 2: continue

            x_old = np.linspace(0, 1, len(vel_seg))
            x_new = np.linspace(0, 1, 100)
            vel_norm = np.interp(x_new, x_old, vel_seg)
            all_profiles.append(vel_norm)
            ax.plot(x_new * 100, vel_norm, color='gray', alpha=0.2)

        if all_profiles:
            mean_profile = np.mean(all_profiles, axis=0)
            std_profile = np.std(all_profiles, axis=0)
            ax.plot(x_new * 100, mean_profile, color='red', linewidth=2.5)
            ax.fill_between(x_new * 100, mean_profile - std_profile, mean_profile + std_profile, color='red', alpha=0.1)

        ax.set_title(f"{phase_name} Phase")
        ax.set_xlabel("% Phase")
        if i == 0: ax.set_ylabel("Velocity (mm/s)")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Time-Normalized Velocity Profiles", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_2d_trajectories(csv_files, phase_data, fs, output_path):
    print("  4/10: 2D Trajectories...")
    num_files = min(len(csv_files), 25)
    cols = 5; rows = math.ceil(num_files / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 18), constrained_layout=True)
    axes = axes.flatten()
    colors = {'X': 'red', 'Y': 'green', 'Z': 'blue'}

    for i, filepath in enumerate(csv_files[:num_files]):
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
        ax.set_title(filename, fontsize=9)
        if i == 0: ax.legend(loc='upper right', fontsize=8)

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.suptitle("Wrist Position Trajectories (X, Y, Z)", fontsize=16)
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_3d_poses(csv_files, phase_data, output_path):
    print("  5/10: 3D Poses...")
    # 3 key events: Pick (idxs[0]), Drink (idxs[1]), Place (idxs[2])
    poses = {0: [], 1: [], 2: []}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) < 3: continue
        df = load_and_process_data(filepath)
        joints = calculate_virtual_joints(df) if df is not None else None
        if joints is not None:
            # Key event indices: Pick, Drink, Place
            key_indices = [idxs[0], idxs[1], idxs[2]]
            for i, idx in enumerate(key_indices):
                try:
                    s = joints['V_Shoulder'].iloc[idx].values
                    e = joints['V_Elbow'].iloc[idx].values
                    w = joints['V_Wrist'].iloc[idx].values
                    poses[i].append((s-s, e-s, w-s))
                except IndexError: pass

    fig = plt.figure(figsize=(15, 5))
    titles = ["Pick", "Drink", "Place"]
    colors = ['green', 'orange', 'blue']

    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        for (s, e, w) in poses[i]:
            ax.plot([s[0],e[0]], [s[1],e[1]], [s[2],e[2]], color='gray', alpha=0.2)
            ax.plot([e[0],w[0]], [e[1],w[1]], [e[2],w[2]], color=colors[i], alpha=0.5, linewidth=2)
            ax.scatter(*w, color=colors[i], s=30)
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlim([-600,600]); ax.set_ylim([-600,600]); ax.set_zlim([-600,600])

    plt.suptitle("Arm Postures at Key Events", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_boxplots(features_df, output_path):
    print("  6/10: Boxplots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['Duration', 'PeakVel', 'PathLen']
    titles = ['Duration (s)', 'Peak Velocity (mm/s)', 'Path Length (mm)']
    phases = ['Pick', 'Drink', 'Place']
    colors = ['lightgreen', 'moccasin', 'lightblue']

    for i, metric in enumerate(metrics):
        ax = axes[i]
        data = [features_df[f'{p}_{metric}'].dropna().values for p in phases if f'{p}_{metric}' in features_df.columns]
        if data:
            bplot = ax.boxplot(data, patch_artist=True, tick_labels=phases[:len(data)])
            for patch, color in zip(bplot['boxes'], colors): patch.set_facecolor(color)
        ax.set_title(titles[i], fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_3d_phase_paths(csv_files, phase_data, output_path):
    print("  7/10: 3D Phase Paths...")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    phase_colors = ['green', 'orange', 'blue']
    phase_labels = ['Pick', 'Drink', 'Place']

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) < 3: continue
        df = load_and_process_data(filepath)
        joints = calculate_virtual_joints(df) if df is not None else None
        if joints is not None:
            s_pos = joints['V_Shoulder'].values
            w_pos = joints['V_Wrist'].values
            rel_w = w_pos - s_pos
            # 3-point indices: Pick (0→idxs[0]), Drink (idxs[0]→idxs[1]), Place (idxs[1]→idxs[2])
            ranges = [(0, idxs[0]), (idxs[0], idxs[1]), (idxs[1], idxs[2])]
            for i, (start, end) in enumerate(ranges):
                if end > start and len(rel_w[start:end]) > 1:
                    ax.plot(rel_w[start:end,0], rel_w[start:end,1], rel_w[start:end,2], color=phase_colors[i], alpha=0.2)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, label=l) for c, l in zip(phase_colors, phase_labels)]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xlim([-600,600]); ax.set_ylim([-600,600]); ax.set_zlim([-600,600])
    plt.title("3D Phase Paths (Shoulder-relative)")
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_cyclograms(cyclograms, output_path):
    print("  8/10: Coordination Cyclograms...")
    fig, ax = plt.subplots(figsize=(10, 8))
    for s_ang, e_ang, fname in cyclograms:
        ax.plot(s_ang, e_ang, alpha=0.3, linewidth=1)
    ax.set_xlabel("Shoulder Elevation (deg)")
    ax.set_ylabel("Elbow Flexion (deg)")
    ax.set_title("Inter-Joint Coordination: Shoulder vs Elbow")
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_advanced_boxplots(adv_features_df, output_path):
    print("  9/10: Advanced Feature Boxplots...")
    metrics = ['Elbow_ROM_Pick', 'Trunk_Compensation', 'Straightness_Index',
               'Drink_Smoothness_LDLJ']
    titles = ['Elbow ROM (deg)', 'Trunk Comp (mm)', 'Straightness (>1=Curved)',
              'Smoothness (LDLJ)']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        if metric in adv_features_df.columns:
            vals = adv_features_df[metric].values
            ax.boxplot(vals, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            x = np.random.normal(1, 0.04, size=len(vals))
            ax.plot(x, vals, 'r.', alpha=0.5)
        ax.set_title(titles[i], fontweight='bold')
        ax.grid(True, axis='y', linestyle='--')
    plt.suptitle("Advanced Kinematic Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_efficiency_scatter(adv_features_df, output_path):
    print("  10/10: Efficiency Scatter...")
    plt.figure(figsize=(10, 6))
    plt.scatter(adv_features_df['Straightness_Index'], adv_features_df['Efficiency_Ratio'],
                c=adv_features_df['Trunk_Compensation'], cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(label='Trunk Compensation (mm)')
    plt.xlabel("Straightness (1.0 = Perfect Line)")
    plt.ylabel("Efficiency Ratio")
    plt.title("Efficiency Analysis")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()

# --- Main ---

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))

    parser = argparse.ArgumentParser(description="Advanced Kinematic Analysis (10 outputs).")
    parser.add_argument("--dataset", type=str, choices=['healthy', 'stroke'], default='stroke')
    args = parser.parse_args()

    if args.dataset == 'healthy':
        data_dir = os.path.join(project_root, "data", "kinematic", "Healthy")
        output_dir = os.path.join(script_dir, "healthy")
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "healthy_phase_indices.json")
    else:
        data_dir = os.path.join(project_root, "data", "kinematic", "Stroke")
        output_dir = script_dir
        json_path = os.path.join(script_dir, "stroke_phase_indices.json")

    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run new_interactive_phase_selector.py first.")
        return

    with open(json_path, 'r') as f:
        phase_data = json.load(f)

    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"\nAdvanced Analysis: {args.dataset.upper()} ({len(csv_files)} files)")
    print("=" * 50)

    fs = 200.0

    # Extract features
    features_df = extract_features(csv_files, phase_data, fs)
    adv_features_df, cyclograms = extract_advanced_features(csv_files, phase_data, fs)

    if features_df.empty:
        print("No valid data. Check phase selection (need 3 points per file).")
        return

    # Generate 10 outputs
    generate_heatmap(features_df, os.path.join(output_dir, "analysis_01_heatmap.png"))
    generate_parallel_coordinates(features_df, os.path.join(output_dir, "analysis_02_parallel.png"))
    generate_time_normalized_plot(csv_files, phase_data, fs, os.path.join(output_dir, "analysis_03_velocity.png"))
    generate_2d_trajectories(csv_files, phase_data, fs, os.path.join(output_dir, "analysis_04_trajectories.png"))
    generate_3d_poses(csv_files, phase_data, os.path.join(output_dir, "analysis_05_poses.png"))
    generate_boxplots(features_df, os.path.join(output_dir, "analysis_06_boxplots.png"))
    generate_3d_phase_paths(csv_files, phase_data, os.path.join(output_dir, "analysis_07_paths3d.png"))
    generate_cyclograms(cyclograms, os.path.join(output_dir, "analysis_08_cyclograms.png"))
    generate_advanced_boxplots(adv_features_df, os.path.join(output_dir, "analysis_09_advanced.png"))
    generate_efficiency_scatter(adv_features_df, os.path.join(output_dir, "analysis_10_efficiency.png"))

    print("=" * 50)
    print(f"All 10 outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
