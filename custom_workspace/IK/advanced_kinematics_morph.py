import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob
import re

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

# POINT THIS TO YOUR SMOOTHED DATA
AUGMENTED_DIR = os.path.join(root_data, "data/kinematic/augmented") 
STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
OUTPUT_DIR = os.path.join(AUGMENTED_DIR, "advanced_analysis")

REF_COLS = ["sh", "shoulder", "acromion"]
WR_COLS = ["wr_", "wrist"]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Helpers ---

def normalize_wrist(df):
    """Extracts Wrist relative to Shoulder, Normalize to Start at 0."""
    cols = {c.lower(): c for c in df.columns}
    
    # Identify columns
    sh_c = [cols[c] for c in cols if any(r in c for r in REF_COLS)]
    wr_c = [cols[c] for c in cols if any(w in c for w in WR_COLS) and 'vec' not in c]
    
    if len(sh_c) < 3 or len(wr_c) < 3: return None
    sh_c.sort(); wr_c.sort()
    
    # Relative Position
    rel = pd.DataFrame()
    rel['x'] = df[wr_c[0]] - df[sh_c[0]]
    rel['y'] = df[wr_c[1]] - df[sh_c[1]]
    rel['z'] = df[wr_c[2]] - df[sh_c[2]]
    
    # Unit Scaling
    if (rel.max() - rel.min()).mean() > 50: rel /= 1000.0
        
    return rel - rel.iloc[0] # Zero start

def get_kinematics(df):
    """Returns displacement, velocity profile, and endpoint."""
    # 1. Displacement Profile
    disp = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    # 2. Velocity Profile
    vel = np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2) * 100 
    
    # 3. Tortuosity
    arc_len = np.sum(np.sqrt(np.diff(df['x'])**2 + np.diff(df['y'])**2 + np.diff(df['z'])**2))
    chord_len = np.linalg.norm(df.iloc[-1] - df.iloc[0])
    
    # SAFETY CHECK: Avoid divide by zero
    if chord_len < 0.001: 
        tortuosity = 1.0 # If didn't move, it's "perfectly straight" (technically undefined but 1.0 is safe)
    else:
        tortuosity = arc_len / chord_len
        
    return disp, vel, df.iloc[-1], tortuosity

def clean_data(data_list):
    """Removes NaN/Inf values from a list."""
    arr = np.array(data_list)
    return arr[np.isfinite(arr)]

# --- Plotting Functions ---

def plot_velocity_spaghetti(data_map):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ['Stroke (Original)', 'Augmented (Generated)', 'Healthy (Target)']
    keys = ['Stroke', 'Augmented', 'Healthy']
    colors = ['red', 'blue', 'green']
    
    for ax, key, color, title in zip(axes, keys, colors, titles):
        dataset = data_map[key]
        if len(dataset) > 100:
            import random
            dataset = random.sample(dataset, 100)
            
        for item in dataset:
            vel = item['vel']
            if len(vel) == 0: continue
            t = np.linspace(0, 100, len(vel))
            ax.plot(t, vel, color=color, alpha=0.1)
            
        ax.set_title(title)
        ax.set_xlabel("% Movement Time")
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel("Velocity (m/s)")
    plt.suptitle("Global Velocity Profile Shapes", fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "1_velocity_bell_curves.png"), dpi=150)
    plt.close()

def plot_phase_portraits(data_map):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    keys = ['Stroke', 'Augmented', 'Healthy']
    
    for ax, key in zip(axes, keys):
        dataset = data_map[key]
        if len(dataset) > 80: dataset = dataset[:80]
        
        for item in dataset:
            disp = item['disp'][:-1]
            vel = item['vel']
            if len(disp) != len(vel): continue # Safety skip
            
            c = 'red' if key=='Stroke' else 'green'
            if key == 'Augmented':
                # Use cool colormap for augmented
                try: cmap = matplotlib.colormaps['cool']
                except: cmap = plt.cm.cool
                score = item.get('score', 20)
                norm_score = (score - 20) / 46
                c = cmap(norm_score)
                
            ax.plot(disp, vel, color=c, alpha=0.3, linewidth=1)
            
        ax.set_title(f"{key} Phase Plane")
        ax.set_xlabel("Displacement (m)")
        ax.grid(True, alpha=0.3)
        
    axes[0].set_ylabel("Velocity (m/s)")
    plt.suptitle("Phase Portraits (Dynamics)", fontsize=16)
    plt.savefig(os.path.join(OUTPUT_DIR, "2_phase_portraits.png"), dpi=150)
    plt.close()

def plot_endpoints(data_map):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Helper to plot cleaned data
    def safe_scatter(data, color, label, size=20, alpha=0.5):
        if not data: return
        ends = np.array([d['end'] for d in data])
        # Check for NaNs
        mask = np.isfinite(ends).all(axis=1)
        ends = ends[mask]
        if len(ends) > 0:
            ax.scatter(ends[:,0], ends[:,1], ends[:,2], c=color, alpha=alpha, label=label, s=size)

    safe_scatter(data_map['Stroke'], 'red', 'Stroke Endpoints')
    safe_scatter(data_map['Healthy'], 'green', 'Healthy Targets')
    safe_scatter(data_map['Augmented'], 'blue', 'Augmented Cloud', size=10, alpha=0.1)
    
    ax.set_title("3D Endpoint Distribution")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "3_endpoint_accuracy.png"), dpi=150)
    plt.close()

def plot_tortuosity(data_map):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # CLEAN DATA BEFORE PLOTTING
    s_tort = clean_data([d['tort'] for d in data_map['Stroke']])
    h_tort = clean_data([d['tort'] for d in data_map['Healthy']])
    a_tort = clean_data([d['tort'] for d in data_map['Augmented']])
    
    # Filter out extreme outliers (e.g. tortuosity > 10 is likely bad data)
    s_tort = s_tort[s_tort < 10]
    h_tort = h_tort[h_tort < 10]
    a_tort = a_tort[a_tort < 10]
    
    bins = np.linspace(1, 5, 50) 
    
    if len(s_tort) > 0: ax.hist(s_tort, bins, alpha=0.5, label='Stroke', color='red', density=True)
    if len(h_tort) > 0: ax.hist(h_tort, bins, alpha=0.5, label='Healthy', color='green', density=True)
    if len(a_tort) > 0: ax.hist(a_tort, bins, alpha=0.3, label='Augmented', color='blue', density=True)
    
    ax.set_title("Path Tortuosity (1.0 = Straight)")
    ax.set_xlabel("Tortuosity Index")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_path_curvature.png"), dpi=150)
    plt.close()

# --- Main ---

def main():
    print("--- Gathering Data for Advanced Report ---")
    data_map = {'Stroke': [], 'Healthy': [], 'Augmented': []}
    
    # 1. Load Stroke
    for f in glob.glob(os.path.join(STROKE_DIR, "*.csv")):
        try:
            df = pd.read_csv(f)
            norm = normalize_wrist(df)
            if norm is not None:
                d, v, end, t = get_kinematics(norm)
                data_map['Stroke'].append({'disp': d, 'vel': v, 'end': end, 'tort': t})
        except: continue
            
    # 2. Load Healthy
    for f in glob.glob(os.path.join(HEALTHY_DIR, "*.csv")):
        try:
            df = pd.read_csv(f)
            norm = normalize_wrist(df)
            if norm is not None:
                d, v, end, t = get_kinematics(norm)
                data_map['Healthy'].append({'disp': d, 'vel': v, 'end': end, 'tort': t})
        except: continue
            
    # 3. Load Augmented
    aug_folders = glob.glob(os.path.join(AUGMENTED_DIR, "*_to_*"))
    print(f"Scanning {len(aug_folders)} augmented pairs...")
    
    for folder in aug_folders:
        for f in glob.glob(os.path.join(folder, "FMA_*.csv")):
            try:
                score = int(re.search(r'FMA_(\d+)', os.path.basename(f)).group(1))
                df = pd.read_csv(f)
                norm = normalize_wrist(df) 
                if norm is None:
                    # Fallback for generated files which might already be normalized
                    try: norm = df[['Wr_x','Wr_y','Wr_z']].rename(columns={'Wr_x':'x','Wr_y':'y','Wr_z':'z'})
                    except: continue
                
                d, v, end, t = get_kinematics(norm)
                data_map['Augmented'].append({'disp': d, 'vel': v, 'end': end, 'tort': t, 'score': score})
            except: continue

    print(f"Data Loaded: S={len(data_map['Stroke'])}, H={len(data_map['Healthy'])}, A={len(data_map['Augmented'])}")
    
    print("Generating Graphs...")
    plot_velocity_spaghetti(data_map)
    plot_phase_portraits(data_map)
    plot_endpoints(data_map)
    plot_tortuosity(data_map)
    
    print(f"Done! Advanced visualizations saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()