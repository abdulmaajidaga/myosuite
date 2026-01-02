import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for server running
import matplotlib.pyplot as plt
import os
import glob
import re

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
# IMPORTANT: Point this to your SMOOTHED data folder
AUGMENTED_DIR = os.path.join(root_data, "data/kinematic/augmented") 
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

# Keywords to find Shoulder and Wrist columns
REF_COLS = ["sh", "shoulder", "acromion"]
WR_COLS_SUB = ["wr_", "wrist"]

# --- Helpers ---

def normalize_trajectory(df):
    """
    1. Finds Shoulder and Wrist columns.
    2. Calculates Wrist position RELATIVE to Shoulder.
    3. Normalizes units to meters.
    4. Translates start position to (0,0,0).
    Returns DataFrame with 'x', 'y', 'z'.
    """
    cols = {c.lower(): c for c in df.columns}
    
    # Find cols based on keywords
    sh_c = [cols[c] for c in cols if any(r in c for r in REF_COLS)]
    wr_c = [cols[c] for c in cols if any(w in c for w in WR_COLS_SUB) and 'vec' not in c]
    
    # Need at least 3 of each (x,y,z)
    if len(sh_c) < 3 or len(wr_c) < 3:
        return None
        
    sh_c.sort(); wr_c.sort()
    
    # Calculate Relative Position (Wrist - Shoulder)
    rel = pd.DataFrame()
    rel['x'] = df[wr_c[0]] - df[sh_c[0]]
    rel['y'] = df[wr_c[1]] - df[sh_c[1]]
    rel['z'] = df[wr_c[2]] - df[sh_c[2]]
    
    # Unit Check: If range is large (>50), assume mm and convert to m
    if (rel.max() - rel.min()).mean() > 50:
        rel /= 1000.0
        
    # Zero Start: Force all paths to start at the origin
    return rel - rel.iloc[0]

# --- Main Visualizer ---

def main():
    print(f"--- Starting Global Spatial Trend Visualization ---")
    print(f"Reading from: {AUGMENTED_DIR}")
    
    # Setup Figure: 3 subplots for Top, Side, Front views
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131) # Top View (X-Y)
    ax2 = fig.add_subplot(132) # Side View (X-Z)
    ax3 = fig.add_subplot(133) # Front View (Y-Z)
    
    axes = [ax1, ax2, ax3]
    # Mapping axes to dataframe columns
    labels = [('x', 'y'), ('x', 'z'), ('y', 'z')]
    
    # Formatting axes
    for ax, (xl, yl) in zip(axes, labels):
        ax.set_xlabel(xl.upper() + " (meters)")
        ax.set_ylabel(yl.upper() + " (meters)")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', lw=0.5)
        ax.axvline(0, color='black', lw=0.5)
        # Set consistent limits to make comparison easier
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)

    ax1.set_title("Top-Down View (X-Y)\n(Forward vs Left/Right)")
    ax2.set_title("Side View (X-Z)\n(Forward vs Up/Down)")
    ax3.set_title("Front View (Y-Z)\n(Left/Right vs Up/Down)")

    # 1. Plot Stroke (Red, high transparency)
    print("Plotting Stroke data...")
    stroke_files = glob.glob(os.path.join(STROKE_DIR, "*.csv"))
    for f in stroke_files:
        df = pd.read_csv(f)
        norm = normalize_trajectory(df)
        if norm is not None:
            for ax, (xc, yc) in zip(axes, labels):
                ax.plot(norm[xc], norm[yc], c='red', alpha=0.05, lw=1)
                
    # 2. Plot Healthy (Green, high transparency)
    print("Plotting Healthy data...")
    healthy_files = glob.glob(os.path.join(HEALTHY_DIR, "*.csv"))
    for f in healthy_files:
        df = pd.read_csv(f)
        norm = normalize_trajectory(df)
        if norm is not None:
            for ax, (xc, yc) in zip(axes, labels):
                ax.plot(norm[xc], norm[yc], c='green', alpha=0.05, lw=1)

    # 3. Plot Augmented (Color Gradient based on FMA)
    print("Plotting Augmented data...")
    pair_folders = glob.glob(os.path.join(AUGMENTED_DIR, "*_to_*"))
    
    # --- FIX START ---
    # Replaced deprecated matplotlib.cm.get_cmap with matplotlib.colormaps
    try:
        cmap = matplotlib.colormaps['cool'] 
    except AttributeError:
        # Fallback for very old matplotlib versions just in case
        cmap = plt.cm.cool
    # --- FIX END ---
    
    norm_fma = matplotlib.colors.Normalize(vmin=20, vmax=66)

    count = 0
    for folder in pair_folders:
        aug_files = glob.glob(os.path.join(folder, "FMA_*.csv"))
        for f in aug_files:
            try:
                # Extract FMA score from filename
                score = int(re.search(r'FMA_(\d+)', os.path.basename(f)).group(1))
                # Get color based on score
                color = cmap(norm_fma(score))
            except: continue
            
            df = pd.read_csv(f)
            norm = normalize_trajectory(df)
            if norm is not None:
                count += 1
                # Higher FMA = more opaque (easier to see convergence)
                a = 0.02 + 0.1 * norm_fma(score) 
                for ax, (xc, yc) in zip(axes, labels):
                    ax.plot(norm[xc], norm[yc], c=color, alpha=a, lw=1)
    print(f"Plotted {count} augmented trajectories.")

    # Add Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Stroke (Original)'),
        Line2D([0], [0], color='green', lw=2, label='Healthy (Target)'),
        Line2D([0], [0], color=cmap(0.0), lw=2, label='Augmented (FMA ~20)'),
        Line2D([0], [0], color=cmap(1.0), lw=2, label='Augmented (FMA ~66)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right')

    # Save Figure
    output_path = os.path.join(AUGMENTED_DIR, "global_spatial_trends.png")
    plt.tight_layout()
    # High DPI for clarity with many lines
    plt.savefig(output_path, dpi=300)
    print(f"Global spatial trends plot saved to: {output_path}")

if __name__ == "__main__":
    main()