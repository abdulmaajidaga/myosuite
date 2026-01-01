import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

# REPLACE THESE PATHS with actual files from your dataset to test
FILE_STROKE   = os.path.join(root_data, "data/kinematic/Stroke/processed/S11_12_1_processed.csv")
FILE_HEALTHY  = os.path.join(root_data, "data/kinematic/Healthy/processed/16_12_3_processed.csv")
# Pick a generated file that corresponds to the middle (e.g., FMA 40)
FILE_GEN      = os.path.join(root_data, "data/kinematic/augmented/S11_12_1_to_16_12_3_processed/FMA_45.csv")

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        return None
    df = pd.read_csv(path)
    
    # Standardize column names if needed (handle case sensitivity)
    df.columns = [c.lower() for c in df.columns]
    
    # Extract Wrist coordinates (we focus on End-Effector for quality analysis)
    # Adjust 'wr_x' if your columns are named differently (e.g. 'wrist_x')
    try:
        data = df[['wr_x', 'wr_y', 'wr_z']].copy()
        data.columns = ['x', 'y', 'z']
        return data
    except KeyError:
        print(f"Error: Could not find 'wr_x/y/z' columns in {os.path.basename(path)}")
        return None

def calc_kinematics(df, fs=100):
    """
    Calculates velocity, acceleration, and jerk.
    Assumes sampling frequency (fs) of roughly 100Hz (standard for mocap).
    """
    # 1. Displacement
    disp = np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2)
    
    # 2. Velocity (Scalar Speed)
    # dist / time (dt = 1/fs)
    vel = disp * fs
    
    # 3. Acceleration (Change in velocity)
    acc = np.diff(vel) * fs
    
    # 4. Jerk (Change in acceleration) - Metric for Smoothness
    jerk = np.diff(acc) * fs
    
    # 5. Path Efficiency Ratio (Straight Line vs Actual Path)
    start = df.iloc[0][['x','y','z']]
    end   = df.iloc[-1][['x','y','z']]
    straight_dist = np.linalg.norm(end - start)
    actual_path_len = np.sum(disp)
    
    # Avoid div by zero
    efficiency = straight_dist / actual_path_len if actual_path_len > 0 else 0
    
    return {
        'vel_profile': vel,
        'mean_vel': np.mean(vel),
        'max_vel': np.max(vel),
        'rms_jerk': np.sqrt(np.mean(jerk**2)),
        'efficiency': efficiency,
        'duration_frames': len(df)
    }

def plot_comparison(f_stroke, f_gen, f_healthy):
    print("--- Loading Files ---")
    d_stroke  = load_data(f_stroke)
    d_gen     = load_data(f_gen)
    d_healthy = load_data(f_healthy)
    
    if d_stroke is None or d_gen is None or d_healthy is None:
        return

    print("--- Calculating Kinematics ---")
    k_stroke  = calc_kinematics(d_stroke)
    k_gen     = calc_kinematics(d_gen)
    k_healthy = calc_kinematics(d_healthy)

    # --- PLOTTING ---
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Kinematic Analysis: Stroke vs Generated (FMA 45) vs Healthy", fontsize=16)

    # 1. 3D Trajectory Comparison
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(d_stroke['x'], d_stroke['y'], d_stroke['z'], c='red', label='Stroke (Original)', lw=2)
    ax1.plot(d_gen['x'], d_gen['y'], d_gen['z'], c='blue', label='Generated (FMA 45)', lw=2, linestyle='--')
    ax1.plot(d_healthy['x'], d_healthy['y'], d_healthy['z'], c='green', label='Healthy (Target)', lw=2)
    ax1.set_title("Spatial Path Comparison")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend()

    # 2. Velocity Profile Comparison
    ax2 = fig.add_subplot(2, 2, 2)
    # Normalize time axis to % of movement
    t_s = np.linspace(0, 100, len(k_stroke['vel_profile']))
    t_g = np.linspace(0, 100, len(k_gen['vel_profile']))
    t_h = np.linspace(0, 100, len(k_healthy['vel_profile']))
    
    ax2.plot(t_s, k_stroke['vel_profile'], c='red', alpha=0.6, label='Stroke')
    ax2.plot(t_g, k_gen['vel_profile'], c='blue', lw=2, label='Generated')
    ax2.plot(t_h, k_healthy['vel_profile'], c='green', alpha=0.6, label='Healthy')
    ax2.set_title("Velocity Profile (Normalized Time)")
    ax2.set_xlabel("% of Movement")
    ax2.set_ylabel("Speed (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Bar Chart: Smoothness (Jerk) - Lower is Better
    ax3 = fig.add_subplot(2, 3, 4)
    jerks = [k_stroke['rms_jerk'], k_gen['rms_jerk'], k_healthy['rms_jerk']]
    colors = ['red', 'blue', 'green']
    ax3.bar(['Stroke', 'Gen', 'Healthy'], jerks, color=colors, alpha=0.7)
    ax3.set_title("Smoothness (RMS Jerk)\nLower is Better")
    ax3.set_ylabel("Jerk Magnitude")

    # 4. Bar Chart: Efficiency - Higher (closer to 1.0) is Better
    ax4 = fig.add_subplot(2, 3, 5)
    effs = [k_stroke['efficiency'], k_gen['efficiency'], k_healthy['efficiency']]
    ax4.bar(['Stroke', 'Gen', 'Healthy'], effs, color=colors, alpha=0.7)
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax4.set_title("Path Efficiency (0-1)\nHigher is Better")
    ax4.set_ylim(0, 1.1)

    # 5. Bar Chart: Max Speed - Healthy is usually faster
    ax5 = fig.add_subplot(2, 3, 6)
    speeds = [k_stroke['max_vel'], k_gen['max_vel'], k_healthy['max_vel']]
    ax5.bar(['Stroke', 'Gen', 'Healthy'], speeds, color=colors, alpha=0.7)
    ax5.set_title("Max Speed (m/s)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # You can pass files via command line or use variables at top
    if len(sys.argv) == 4:
        plot_comparison(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Use hardcoded paths for testing
        plot_comparison(FILE_STROKE, FILE_GEN, FILE_HEALTHY)