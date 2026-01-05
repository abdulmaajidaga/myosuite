import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter, resample
import re

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Paths
AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Augmented")
REAL_HEALTHY_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Healthy/processed")
REAL_STROKE_DIR = os.path.join(PROJECT_ROOT, "data/kinematic/Stroke/processed")
SCORES_FILE = os.path.join(BASE_DIR, "output/scores.csv")

# Constants
DT = 0.02  # Time step (Assuming 50Hz)
PLOT_POINTS = 100 # Normalize all movements to 0-100% progress

def load_score_map():
    try:
        df = pd.read_csv(SCORES_FILE)
        id_col = df.columns[0]
        score_col = df.columns[1]
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '').str.strip()
        return dict(zip(df[id_col], df[score_col]))
    except:
        return {}

def calculate_wrist_velocity(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # robust column check
        cols = {c.lower(): c for c in df.columns}
        if not all(k in cols for k in ['wr_x', 'wr_y', 'wr_z']):
            return None

        # Extract position
        wx = df[cols['wr_x']].values
        wy = df[cols['wr_y']].values
        wz = df[cols['wr_z']].values

        # Calculate Velocity (differentiation)
        dx = np.diff(wx) / DT
        dy = np.diff(wy) / DT
        dz = np.diff(wz) / DT
        
        # Magnitude
        velocity = np.sqrt(dx**2 + dy**2 + dz**2)

        # Smooth signal (Window 11, Poly 3 is standard for human motion)
        # Check length to avoid errors on tiny files
        if len(velocity) > 15:
            velocity = savgol_filter(velocity, 11, 3)
        
        return velocity
    except:
        return None

def load_velocities(directory, label_type, score_map=None):
    velocities = []
    scores = []
    
    files = []
    # Recursively find CSVs
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".csv"):
                files.append(os.path.join(root, filename))
    
    # Randomly sample Augmented to prevent clutter (200 files max)
    if label_type == "Augmented" and len(files) > 200:
        files = np.random.choice(files, 200, replace=False)

    print(f"Processing {label_type} ({len(files)} files)...")

    for f in files:
        name = os.path.basename(f).replace('_processed.csv', '').replace('.csv', '')
        score = None
        
        # Determine Score
        if label_type == "Augmented":
            match = re.search(r'FMA_(\d+)', name)
            if match: score = int(match.group(1))
        elif label_type == "Real Stroke":
            score = score_map.get(name) if score_map else 30
        elif label_type == "Real Healthy":
            score = 66
            
        # Calculate
        vel = calculate_wrist_velocity(f)
        
        # Normalize time to 0-100%
        if vel is not None and len(vel) > 10:
            vel_norm = resample(vel, PLOT_POINTS)
            velocities.append(vel_norm)
            scores.append(score if score else 0)

    return velocities, scores

def main():
    score_map = load_score_map()
    
    print("--- Loading Data ---")
    h_vels, _ = load_velocities(REAL_HEALTHY_DIR, "Real Healthy")
    s_vels, _ = load_velocities(REAL_STROKE_DIR, "Real Stroke", score_map)
    a_vels, a_scores = load_velocities(AUGMENTED_DIR, "Augmented")
    
    # Split Augmented into Bands for clarity
    a_low = [v for v, s in zip(a_vels, a_scores) if s < 35]
    a_mid = [v for v, s in zip(a_vels, a_scores) if 35 <= s < 55]
    a_high = [v for v, s in zip(a_vels, a_scores) if s >= 55]

    # --- PLOTTING ---
    plt.figure(figsize=(14, 8))
    
    def plot_band(data, color, label, linestyle='-'):
        if not data: return
        arr = np.array(data)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        t = np.linspace(0, 100, len(mean))
        
        plt.plot(t, mean, color=color, label=label, linewidth=2.5, linestyle=linestyle)
        plt.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

    # 1. Healthy (Green - Solid)
    plot_band(h_vels, 'green', 'Real Healthy (FMA 66)')
    
    # 2. Stroke (Red - Solid)
    plot_band(s_vels, 'red', 'Real Stroke (FMA < 30)')
    
    # 3. Augmented (Blue Gradients - Dashed)
    plot_band(a_low, 'darkblue', 'Augmented (FMA 20-35)', '--')
    plot_band(a_mid, 'royalblue', 'Augmented (FMA 35-55)', '--')
    plot_band(a_high, 'skyblue', 'Augmented (FMA 55+)', '--')

    plt.title("Wrist Velocity Profiles: Evolution of Smoothness", fontsize=16)
    plt.xlabel("Movement Progress (%)", fontsize=12)
    plt.ylabel("Velocity (mm/s)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(BASE_DIR, "output/velocity_profiles.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()