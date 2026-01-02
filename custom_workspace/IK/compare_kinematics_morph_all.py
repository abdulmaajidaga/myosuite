import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import glob
import re

# --- Configuration ---
base_dir = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(base_dir)

STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke/processed")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy/processed")
AUGMENTED_DIR = os.path.join(root_data, "data/kinematic/augmented_smooth")
SCORES_FILE = os.path.join(base_dir, "output/scores.csv")

REF_COLS = ["sh", "shoulder", "acromion"]

# --- Helpers ---

def load_scores_map(filepath):
    try:
        df = pd.read_csv(filepath)
        id_col, score_col = df.columns[0], df.columns[1]
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        return dict(zip(df[id_col], df[score_col]))
    except: return {}

def normalize_for_comparison(df):
    """Normalize Wrist relative to Shoulder to ensure fair comparison."""
    cols = {c.lower(): c for c in df.columns}
    sh_cols = [cols[c] for c in cols if any(r in c for r in REF_COLS)]
    wr_cols = [cols[c] for c in cols if 'wr_' in c and 'vec' not in c]
    
    if len(sh_cols) < 3 or len(wr_cols) < 3:
        num_df = df.select_dtypes(include=[np.number])
        return num_df - num_df.iloc[0]
        
    sh_cols.sort(); wr_cols.sort()
    rel_data = pd.DataFrame()
    rel_data['Wr_x'] = df[wr_cols[0]] - df[sh_cols[0]]
    rel_data['Wr_y'] = df[wr_cols[1]] - df[sh_cols[1]]
    rel_data['Wr_z'] = df[wr_cols[2]] - df[sh_cols[2]]
    return rel_data - rel_data.iloc[0]

def calc_metrics(df, group_type, fma_score, filename):
    cols = {c.lower(): c for c in df.columns}
    try:
        x, y, z = df[cols['wr_x']], df[cols['wr_y']], df[cols['wr_z']]
    except KeyError: return None

    # Kinematics
    disp = np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2)
    fs = 100 
    vel = disp * fs
    acc = np.diff(vel) * fs
    jerk = np.diff(acc) * fs
    
    # Efficiency
    start = np.array([x.iloc[0], y.iloc[0], z.iloc[0]])
    end = np.array([x.iloc[-1], y.iloc[-1], z.iloc[-1]])
    straight_dist = np.linalg.norm(end - start)
    path_len = np.sum(disp)
    eff = straight_dist / path_len if path_len > 0 else 0
    
    return {
        'Filename': filename,
        'Group': group_type, # 'Stroke', 'Augmented', 'Healthy'
        'FMA': fma_score,
        'Max_Vel': np.max(vel),
        'Mean_Vel': np.mean(vel),
        'RMS_Jerk': np.sqrt(np.mean(jerk**2)),
        'Efficiency': eff
    }

def find_file(directory, partial_name):
    files = glob.glob(os.path.join(directory, "*.csv"))
    for f in files:
        if partial_name in os.path.basename(f): return f
    return None

# --- Visualization Logic ---

def generate_individual_report(df_res, output_path, title):
    """Generates the per-patient trend lines (RESTORED LOGIC)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Recovery Trends: {title}", fontsize=14)
    
    # 1. Efficiency
    axes[0].plot(df_res['FMA'], df_res['Efficiency'], 'o-', c='purple')
    axes[0].set_title('Path Efficiency (Higher is Better)')
    axes[0].set_xlabel('FMA Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True)
    
    # 2. Jerk
    axes[1].plot(df_res['FMA'], df_res['RMS_Jerk'], 'o-', c='orange')
    axes[1].set_title('RMS Jerk (Lower is Better)')
    axes[1].set_xlabel('FMA Score')
    axes[1].grid(True)
    
    # 3. Velocity
    axes[2].plot(df_res['FMA'], df_res['Max_Vel'], 'o-', c='blue')
    axes[2].set_title('Max Velocity')
    axes[2].set_xlabel('FMA Score')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_global_dashboard(df_all, output_path):
    """Creates a massive comparison plot of ALL data points."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    
    c_map = {'Stroke': 'red', 'Augmented': 'blue', 'Healthy': 'green'}
    alpha_map = {'Stroke': 0.8, 'Augmented': 0.1, 'Healthy': 0.8}
    
    # --- Row 1: Scatter Clouds ---
    
    # 1. Efficiency
    ax1 = plt.subplot(gs[0, 0])
    for g in ['Augmented', 'Stroke', 'Healthy']:
        subset = df_all[df_all['Group'] == g]
        ax1.scatter(subset['FMA'], subset['Efficiency'], 
                   c=c_map[g], alpha=alpha_map[g], label=g, s=15)
    ax1.set_title("Path Efficiency Distribution", fontsize=14)
    ax1.set_ylabel("Efficiency (0-1)")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Jerk
    ax2 = plt.subplot(gs[0, 1])
    for g in ['Augmented', 'Stroke', 'Healthy']:
        subset = df_all[df_all['Group'] == g]
        ax2.scatter(subset['FMA'], subset['RMS_Jerk'], 
                   c=c_map[g], alpha=alpha_map[g], s=15)
    ax2.set_title("Smoothness (RMS Jerk)", fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Velocity
    ax3 = plt.subplot(gs[0, 2])
    for g in ['Augmented', 'Stroke', 'Healthy']:
        subset = df_all[df_all['Group'] == g]
        ax3.scatter(subset['FMA'], subset['Max_Vel'], 
                   c=c_map[g], alpha=alpha_map[g], s=15)
    ax3.set_title("Maximum Velocity", fontsize=14)
    ax3.set_ylabel("Speed (m/s)")
    ax3.grid(True, alpha=0.3)

    # --- Row 2: Box Plots ---
    
    bins = [0, 20, 30, 40, 50, 60, 70]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '66 (Healthy)']
    df_all['FMA_Bin'] = pd.cut(df_all['FMA'], bins=bins, labels=labels)
    
    # 4. Box Plot: Efficiency
    ax4 = plt.subplot(gs[1, 0])
    box_data = []
    box_labels = []
    for label in labels:
        subset = df_all[df_all['FMA_Bin'] == label]['Efficiency'].dropna()
        if len(subset) > 0:
            box_data.append(subset.values)
            box_labels.append(label)
    
    # FIXED: Replaced 'labels=' with 'tick_labels='
    ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                boxprops=dict(facecolor='lightblue'))
    ax4.set_title("Trend: Efficiency per FMA Range", fontsize=14)
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)

    # 5. Box Plot: Jerk
    ax5 = plt.subplot(gs[1, 1])
    box_data = []
    for label in labels:
        subset = df_all[df_all['FMA_Bin'] == label]['RMS_Jerk'].dropna()
        if len(subset) > 0: box_data.append(subset.values)
    
    # FIXED: Replaced 'labels=' with 'tick_labels='
    ax5.boxplot(box_data, tick_labels=box_labels, patch_artist=True, 
                boxprops=dict(facecolor='lightcoral'))
    ax5.set_title("Trend: Jerk per FMA Range", fontsize=14)
    ax5.set_yscale('log')
    ax5.grid(axis='y', alpha=0.3)

    # 6. Text Stats
    ax6 = plt.subplot(gs[1, 2])
    ax6.axis('off')
    corr_eff = df_all['FMA'].corr(df_all['Efficiency'])
    corr_jerk = df_all['FMA'].corr(df_all['RMS_Jerk'])
    count_s = len(df_all[df_all['Group']=='Stroke'])
    count_a = len(df_all[df_all['Group']=='Augmented'])
    count_h = len(df_all[df_all['Group']=='Healthy'])
    
    text_str = (
        f"DATASET SUMMARY\n----------------\n"
        f"Files: {len(df_all)}\n"
        f"  Stroke: {count_s} | Aug: {count_a} | Healthy: {count_h}\n\n"
        f"CORRELATIONS\n----------------\n"
        f"FMA vs Efficiency: {corr_eff:.2f}\n"
        f"FMA vs Smoothness: {corr_jerk:.2f}"
    )
    ax6.text(0.1, 0.5, text_str, fontsize=14, fontfamily='monospace', va='center')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Global Dashboard saved to: {output_path}")
    plt.close()

# --- Main Analysis ---

def main():
    print("--- Starting GLOBAL Kinematic Analysis (Merged Version) ---")
    score_map = load_scores_map(SCORES_FILE)
    pair_folders = glob.glob(os.path.join(AUGMENTED_DIR, "*_to_*"))
    
    global_stats = []

    # 1. Collect Data from Pairs
    for folder in pair_folders:
        pair_id = os.path.basename(folder)
        try:
            parts = pair_id.split('_to_')
            s_id, h_id = parts[0], parts[1]
        except: continue
        
        pair_metrics = [] # List to hold metrics JUST for this folder

        # A. Stroke Data
        s_path = find_file(STROKE_DIR, s_id)
        if s_path:
            start_fma = score_map.get(s_id)
            if not start_fma:
                 for k in score_map: 
                     if k in s_id: start_fma = score_map[k]; break
            start_fma = int(start_fma) if start_fma else 20
            
            df_s = pd.read_csv(s_path)
            df_s_norm = normalize_for_comparison(df_s)
            if (df_s_norm.max() - df_s_norm.min()).mean() > 50: df_s_norm /= 1000.0
            
            m = calc_metrics(df_s_norm, 'Stroke', start_fma, s_id)
            if m: 
                global_stats.append(m)
                pair_metrics.append(m)
            
        # B. Augmented Data
        aug_files = glob.glob(os.path.join(folder, "FMA_*.csv"))
        for f in aug_files:
            try: score = int(re.search(r'FMA_(\d+)', os.path.basename(f)).group(1))
            except: continue
            
            df_aug = pd.read_csv(f)
            m = calc_metrics(df_aug, 'Augmented', score, os.path.basename(f))
            if m: 
                global_stats.append(m)
                pair_metrics.append(m)
            
        # C. Healthy Data
        h_path = find_file(HEALTHY_DIR, h_id)
        if h_path:
            df_h = pd.read_csv(h_path)
            df_h_norm = normalize_for_comparison(df_h)
            if (df_h_norm.max() - df_h_norm.min()).mean() > 50: df_h_norm /= 1000.0
            m = calc_metrics(df_h_norm, 'Healthy', 66, h_id)
            if m: 
                global_stats.append(m)
                pair_metrics.append(m)
        
        # --- D. RESTORED LOGIC: Per-Pair Reporting ---
        # Generate the individual csv and plot for this folder
        if len(pair_metrics) > 2:
            df_pair = pd.DataFrame(pair_metrics).sort_values('FMA')
            
            # Save pair CSV
            df_pair.to_csv(os.path.join(folder, "kinematic_comparison_report.csv"), index=False)
            
            # Save pair Plot
            plot_path = os.path.join(folder, "kinematic_trends.png")
            generate_individual_report(df_pair, plot_path, pair_id)
            # print(f"Saved report for {pair_id}")

    # 2. Generate Global Visualizations
    if global_stats:
        df_all = pd.DataFrame(global_stats)
        
        # Remove duplicates
        df_all = df_all.drop_duplicates(subset=['Filename', 'Group'])
        
        # Save Raw Metrics CSV
        summary_path = os.path.join(AUGMENTED_DIR, "global_kinematic_metrics.csv")
        df_all.to_csv(summary_path, index=False)
        print(f"Metrics CSV saved to: {summary_path}")
        
        # Generate Dashboard
        
        dashboard_path = os.path.join(AUGMENTED_DIR, "global_kinematic_dashboard.png")
        generate_global_dashboard(df_all, dashboard_path)

if __name__ == "__main__":
    main()