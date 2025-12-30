import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import math
from scipy.signal import butter, filtfilt

# --- 1. Data Loading & Processing (Reused logic) ---

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
        # Wrist (Midpoint)
        processed_df[('V_Wrist', 'X')] = (processed_df['WRA']['X'] + processed_df['WRB']['X']) / 2
        processed_df[('V_Wrist', 'Y')] = (processed_df['WRA']['Y'] + processed_df['WRB']['Y']) / 2
        processed_df[('V_Wrist', 'Z')] = (processed_df['WRA']['Z'] + processed_df['WRB']['Z']) / 2
        # Elbow (Midpoint)
        processed_df[('V_Elbow', 'X')] = (processed_df['ELB_L']['X'] + processed_df['ELB_M']['X']) / 2
        processed_df[('V_Elbow', 'Y')] = (processed_df['ELB_L']['Y'] + processed_df['ELB_M']['Y']) / 2
        processed_df[('V_Elbow', 'Z')] = (processed_df['ELB_L']['Z'] + processed_df['ELB_M']['Z']) / 2
        # Shoulder (Centroid)
        processed_df[('V_Shoulder', 'X')] = (processed_df['SA_1']['X'] + processed_df['SA_2']['X'] + processed_df['SA_3']['X']) / 3
        processed_df[('V_Shoulder', 'Y')] = (processed_df['SA_1']['Y'] + processed_df['SA_2']['Y'] + processed_df['SA_3']['Y']) / 3
        processed_df[('V_Shoulder', 'Z')] = (processed_df['SA_1']['Z'] + processed_df['SA_2']['Z'] + processed_df['SA_3']['Z']) / 3
        return processed_df
    except KeyError:
        return None

# --- 2. Advanced Kinematic Calculations ---

def calculate_angles(joints_df):
    """
    Calculates Elbow Flexion and Shoulder Elevation angles.
    """
    s = joints_df['V_Shoulder'].values
    e = joints_df['V_Elbow'].values
    w = joints_df['V_Wrist'].values
    
    # Vectors
    # Upper Arm: Shoulder -> Elbow
    v_upper = e - s
    # Forearm: Elbow -> Wrist
    v_fore = w - e
    # Vertical Vector (assuming Z is Up based on standard Mocap, or we use Y? 
    # play_kinematics uses Z as vertical in plotting)
    v_vert = np.array([0, 0, 1])
    
    angles = {}
    
    # 1. Elbow Flexion (Angle between Upper Arm and Forearm)
    # Cos(theta) = dot(u, v) / (|u| |v|)
    # Note: 180 degrees = Full Extension (straight arm)
    norms_upper = np.linalg.norm(v_upper, axis=1)
    norms_fore = np.linalg.norm(v_fore, axis=1)
    
    dot_prod = np.sum(v_upper * v_fore, axis=1)
    cosine_angle = dot_prod / (norms_upper * norms_fore)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    # Angle in degrees
    # If they point in same direction (straight arm), angle is 0 in vector math?
    # No, S->E and E->W. If straight, they are collinear. Angle is 0. 
    # Usually flexion is measured from 0 (straight) or 180 (straight).
    # Let's use the vector angle: 0 means vectors are aligned (Straight arm).
    elbow_angle = np.degrees(np.arccos(cosine_angle))
    # Anatomically, 0 is often fully flexed and 180 extended, or vice versa.
    # Here, 0 deg = Straight arm (Hyperextension boundary). 90 = Bent.
    angles['Elbow_Flexion'] = elbow_angle

    # 2. Shoulder Elevation (Angle between Upper Arm and Vertical)
    # Using Upper Arm vector (S->E) and Vertical (0,0,1)
    # Note: Usually S->E points DOWN. So angle with (0,0,1) would be ~180.
    # Let's use vector E->S (pointing up to shoulder) vs Vertical? 
    # Or just S->E against Vertical (Down is 180, Up is 0).
    # Abduction implies lifting arm AWAY from torso. 
    # Let's use angle between S->E and Vertical. 
    # 180 = Arm down side. 90 = Arm horizontal. 0 = Arm straight up.
    dot_prod_s = np.sum(v_upper * v_vert, axis=1)
    cos_s = dot_prod_s / norms_upper # |v_vert| is 1
    cos_s = np.clip(cos_s, -1.0, 1.0)
    shoulder_elev = np.degrees(np.arccos(cos_s))
    # Convert to "Elevation": 0 = Down, 90 = Horizontal, 180 = Up
    # Current: 180=Down (vector points -Z vs +Z). 
    # Let's flip it: Elevation = 180 - angle
    angles['Shoulder_Elevation'] = 180 - shoulder_elev
    
    return pd.DataFrame(angles)

def calculate_derivatives(pos, fs):
    """Calc Vel, Acc, Jerk magnitude"""
    vel = np.diff(pos, axis=0) * fs
    acc = np.diff(vel, axis=0) * fs
    jerk = np.diff(acc, axis=0) * fs
    
    # Pad to match length
    vel = np.vstack([np.zeros((1,3)), vel])
    acc = np.vstack([np.zeros((2,3)), acc])
    jerk = np.vstack([np.zeros((3,3)), jerk])
    
    return vel, acc, jerk

def calculate_smoothness(jerk, duration, v_peak):
    """
    Log Dimensionless Jerk
    LDLJ = -ln( (integral(jerk^2) * duration^3) / v_peak^2 ) 
    (Formula variation exists, using a standard one for point-to-point)
    """
    if duration <= 0 or v_peak <= 0: return 0
    
    jerk_sq = np.sum(jerk**2, axis=1) # Magnitude squared
    integral_jerk = np.sum(jerk_sq) * (1/200.0) # dt
    
    # Dimensionless Jerk
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

# --- 3. Main Analysis Function ---

import argparse

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    # Parse Args
    parser = argparse.ArgumentParser(description="Advanced Kinematic Features.")
    parser.add_argument("--dataset", type=str, choices=['healthy', 'stroke'], default='stroke', help="Dataset to process.")
    args = parser.parse_args()
    
    # Determine Paths
    if args.dataset == 'healthy':
        filtered_dir = os.path.join(project_root, "data", "kinematic", "Healthy", "filtered")
        output_dir = os.path.join(script_dir, "healthy")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        json_path = os.path.join(output_dir, "manual_phase_indices.json")
    else:
        filtered_dir = os.path.join(project_root, "data", "kinematic", "Stroke", "filtered")
        output_dir = script_dir
        json_path = os.path.join(script_dir, "manual_phase_indices.json")
    
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Please run interactive_phase_selector.py --dataset {args.dataset} first.")
        return
        
    with open(json_path, 'r') as f:
        phase_data = json.load(f)
        
    csv_files = sorted(glob.glob(os.path.join(filtered_dir, "*.csv")))[:25]
    fs = 200.0
    
    if not csv_files:
        print(f"No CSV files found in {filtered_dir}")
        return

    # Storage for aggregated features
    features = []
    
    # Storage for Cyclograms (Angle-Angle)
    cyclograms = [] # list of (shoulder_angle_array, elbow_angle_array, filename)
    
    print(f"Processing 25 files for advanced features ({args.dataset.upper()})...")
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        idxs = phase_data.get(filename, [])
        if len(idxs) != 3: continue
        
        df = load_and_process_data(filepath)
        if df is None: continue
        joints = calculate_virtual_joints(df)
        if joints is None: continue
        
        # 1. Calc Angles & Kinematics
        angles = calculate_angles(joints)
        w_pos = joints['V_Wrist'].values
        vel, acc, jerk = calculate_derivatives(w_pos, fs)
        vel_mag = np.linalg.norm(vel, axis=1)
        
        # 2. Extract Phase Indices
        # Reach: 0 -> idxs[0]
        # Lift: idxs[0] -> idxs[1]
        # Place: idxs[1] -> idxs[2]
        # Rest: idxs[2] -> end
        
        # --- Feature 1: Joint Angle Features (Reach Phase) ---
        r_start, r_end = 0, idxs[0]
        reach_angles = angles.iloc[r_start:r_end]
        
        # Elbow Extension ROM (Max - Min during reach)
        elbow_rom = reach_angles['Elbow_Flexion'].max() - reach_angles['Elbow_Flexion'].min() if not reach_angles.empty else 0
        
        # Trunk Compensation (Shoulder Displacement)
        s_pos = joints['V_Shoulder'].values
        s_disp = np.linalg.norm(s_pos - s_pos[0], axis=1)
        max_trunk_comp = np.max(s_disp)
        
        # --- Feature 2: Coordination (Whole movement or Reach/Lift) ---
        # Storing for plot
        cyclograms.append((angles['Shoulder_Elevation'].values, angles['Elbow_Flexion'].values, filename))
        
        # --- Feature 3: Efficiency (Reach Phase) ---
        # Hand/Joint Ratio: Dist Wrist / Sum(Abs(Delta Angles))
        reach_w_pos = w_pos[r_start:r_end]
        if len(reach_w_pos) > 1:
            wrist_dist = np.sum(np.linalg.norm(np.diff(reach_w_pos, axis=0), axis=1))
            
            d_shoulder = np.sum(np.abs(np.diff(reach_angles['Shoulder_Elevation'])))
            d_elbow = np.sum(np.abs(np.diff(reach_angles['Elbow_Flexion'])))
            joint_travel = d_shoulder + d_elbow
            
            efficiency = wrist_dist / joint_travel if joint_travel > 0 else 0
            
            # Straightness
            straightness = calculate_straightness(reach_w_pos)
        else:
            efficiency = 0
            straightness = 1.0
        
        # --- Feature 4: Phase Specifics ---
        
        # Reach: Time to Peak Velocity (%)
        if len(vel_mag[r_start:r_end]) > 0:
            peak_idx = np.argmax(vel_mag[r_start:r_end])
            time_to_peak = peak_idx / len(vel_mag[r_start:r_end]) # Normalized 0-1
        else:
            time_to_peak = 0
            
        # Lift: Smoothness (Log Dimensionless Jerk)
        l_start, l_end = idxs[0], idxs[1]
        if l_end > l_start:
            lift_dur = (l_end - l_start) / fs
            lift_vpeak = np.max(vel_mag[l_start:l_end])
            lift_jerk = jerk[l_start:l_end]
            smoothness = calculate_smoothness(lift_jerk, lift_dur, lift_vpeak)
            
            # Lift: Max Shoulder Abduction (Elevation)
            lift_angles = angles.iloc[l_start:l_end]
            max_sh_elev = lift_angles['Shoulder_Elevation'].max() if len(lift_angles) > 0 else 0
        else:
            smoothness = 0
            max_sh_elev = 0
        
        # Place: Target Accuracy (Final Position Variance - handled at group level, 
        # but here we'll store final pos to calc variance later)
        p_start, p_end = idxs[1], idxs[2]
        final_pos = w_pos[p_end] if p_end < len(w_pos) else w_pos[-1]
        
        # Rest: Static Tremor
        rest_start = idxs[2]
        if rest_start < len(w_pos):
            rest_pos = w_pos[rest_start:]
            # Std dev of position (euclidean from mean)
            mean_rest = np.mean(rest_pos, axis=0)
            tremor = np.mean(np.linalg.norm(rest_pos - mean_rest, axis=1))
        else:
            tremor = 0
            
        features.append({
            'Filename': filename,
            'Elbow_ROM_Reach': elbow_rom,
            'Trunk_Compensation': max_trunk_comp,
            'Efficiency_Ratio': efficiency,
            'Straightness_Index': straightness,
            'Time_to_Peak_Vel': time_to_peak,
            'Lift_Smoothness_LDLJ': smoothness,
            'Max_Shoulder_Elev_Lift': max_sh_elev,
            'Rest_Tremor': tremor,
            'Final_X': final_pos[0], 'Final_Y': final_pos[1], 'Final_Z': final_pos[2]
        })

    if not features:
        print("No valid data processed. Check if manual phase selection is complete.")
        return

    df_feats = pd.DataFrame(features)
    
    # Calc Place Accuracy (Group Variance)
    # Distance of each final pos from group mean final pos
    mean_target = df_feats[['Final_X', 'Final_Y', 'Final_Z']].mean().values
    df_feats['Place_Accuracy_Error'] = df_feats.apply(
        lambda r: np.linalg.norm(np.array([r['Final_X'], r['Final_Y'], r['Final_Z']]) - mean_target), axis=1
    )
    
    # --- Generation of Plots ---
    
    # 1. Angle-Angle Plots (Coordination)
    print("Generating Coordination Cyclograms...")
    fig, ax = plt.subplots(figsize=(10, 8))
    for s_ang, e_ang, fname in cyclograms:
        ax.plot(s_ang, e_ang, alpha=0.3, linewidth=1, label=fname)
    ax.set_xlabel("Shoulder Elevation (deg)")
    ax.set_ylabel("Elbow Flexion (deg)")
    ax.set_title("Inter-Joint Coordination: Shoulder vs Elbow\n(Smooth loops = Good coordination)")
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "advanced_1_coordination_cyclograms.png"), dpi=150)
    
    # 2. Feature Distributions (Boxplots)
    print("Generating Feature Boxplots...")
    metrics_to_plot = [
        'Elbow_ROM_Reach', 'Trunk_Compensation', 'Straightness_Index', 
        'Lift_Smoothness_LDLJ', 'Rest_Tremor', 'Place_Accuracy_Error'
    ]
    titles = [
        'Reach: Elbow ROM (deg)', 'Trunk Comp (mm)', 'Reach Straightness (>1=Curved)',
        'Lift Smoothness (LDLJ)', 'Rest Tremor (mm)', 'Place Error (mm)'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        vals = df_feats[metric].values
        ax.boxplot(vals, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax.set_title(titles[i], fontweight='bold')
        ax.grid(True, axis='y', linestyle='--')
        # Add jitter points
        x = np.random.normal(1, 0.04, size=len(vals))
        ax.plot(x, vals, 'r.', alpha=0.5)
        
    plt.suptitle("Advanced Kinematic Features Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "advanced_2_feature_distributions.png"), dpi=150)
    
    # 3. Efficiency Scatter (Straightness vs Efficiency)
    print("Generating Efficiency Scatter...")
    plt.figure(figsize=(10, 6))
    plt.scatter(df_feats['Straightness_Index'], df_feats['Efficiency_Ratio'], c=df_feats['Trunk_Compensation'], cmap='viridis', s=100, alpha=0.8)
    plt.colorbar(label='Trunk Compensation (mm)')
    plt.xlabel("Reach Straightness (1.0 = Perfect Line)")
    plt.ylabel("Hand/Joint Efficiency Ratio")
    plt.title("Efficiency Analysis: Do straighter reaches require less joint work?")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "advanced_3_efficiency_scatter.png"), dpi=150)
    
    print("\nAdvanced Analysis Complete.")
    print(f"1. {os.path.join(output_dir, 'advanced_1_coordination_cyclograms.png')}")
    print(f"2. {os.path.join(output_dir, 'advanced_2_feature_distributions.png')}")
    print(f"3. {os.path.join(output_dir, 'advanced_3_efficiency_scatter.png')}")

if __name__ == "__main__":
    main()
