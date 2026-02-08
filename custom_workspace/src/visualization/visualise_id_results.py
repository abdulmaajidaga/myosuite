import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_DIR = None  # Set via CLI arg or defaults to first available ID output

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def load_data(results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    if results_dir is None:
        # Default: pick the first subject directory in originals/id/
        id_root = get_path("output_originals_id")
        if os.path.isdir(id_root):
            subjects = sorted([d for d in os.listdir(id_root)
                               if os.path.isdir(os.path.join(id_root, d))])
            if subjects:
                results_dir = os.path.join(id_root, subjects[0])
    if results_dir is None:
        print("Error: No ID results directory found.")
        sys.exit(1)
    try:
        act = pd.read_csv(os.path.join(results_dir, 'activations.csv'))
        trq = pd.read_csv(os.path.join(results_dir, 'torques.csv'))
        print(f"Loaded ID results from: {results_dir}")
        return act, trq
    except FileNotFoundError:
        print(f"Error: Could not find 'activations.csv' or 'torques.csv' in {results_dir}")
        sys.exit(1)

def plot_dashboard():
    act, trq = load_data()
    time = act['time']
    
    # Setup Figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle('Neuromechanical Validation Dashboard', fontsize=16, fontweight='bold')

    # ---------------------------------------------------------
    # PLOT 1: The "Mirror" Test (Elbow Flexion)
    # ---------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Torque (Demand)
    ax1.set_title("1. Neuromechanical Consistency (The 'Mirror' Test)", fontsize=12, fontweight='bold')
    ax1.plot(time, trq['elbow_flexion'], color='black', linewidth=2, linestyle='--', label='Net Elbow Torque (Nm)')
    ax1.set_ylabel("Torque (Nm)", fontweight='bold')
    ax1.axhline(0, color='grey', alpha=0.3)
    
    # Muscles (Supply)
    # Summing synergists to show total drive
    flexors = act.filter(regex='BIC|BRA|BRD').sum(axis=1) # Biceps, Brachialis, Brachioradialis
    extensors = act.filter(regex='TRI|ANC').sum(axis=1)   # Triceps, Anconeus
    
    ax1_r = ax1.twinx()
    ax1_r.plot(time, flexors, color='#d62728', alpha=0.8, linewidth=2, label='Total Flexors')
    ax1_r.plot(time, -extensors, color='#1f77b4', alpha=0.8, linewidth=2, label='Total Extensors (Inverted)')
    ax1_r.set_ylabel("Summed Activation (0-1)", fontweight='bold', color='#d62728')
    
    # Legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    # ---------------------------------------------------------
    # PLOT 2: Wrist Stabilization Strategy
    # ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("2. Wrist Stabilization (Cocontraction)", fontsize=12, fontweight='bold')
    
    # Wrist Flexors vs Extensors
    # ECR = Extensor Carpi Radialis, FCR = Flexor Carpi Radialis
    ax2.plot(time, act['ECRB'], label='Extensor (ECRB)', color='purple', linewidth=2)
    ax2.plot(time, act['FCR'], label='Flexor (FCR)', color='orange', linewidth=2)
    ax2.fill_between(time, act['ECRB'], act['FCR'], color='grey', alpha=0.1)
    
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Activation")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # PLOT 3: Muscle Activation Heatmap
    # ---------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("3. Full Muscle Recruitment Pattern", fontsize=12, fontweight='bold')
    
    # Select only interesting muscles (skip tiny ones for clarity)
    key_muscles = [
        'DELT1', 'DELT2', 'BIClong', 'BICshort', 'BRA', 'BRD', 
        'TRIlong', 'TRIlat', 'ECRB', 'ECRL', 'FCR', 'FCU'
    ]
    
    # Transpose for heatmap
    data_subset = act[key_muscles].T
    sns.heatmap(data_subset, cmap='magma', ax=ax3, vmin=0, vmax=1, cbar_kws={'label': 'Activation Level'})
    ax3.set_xlabel("Time Frames")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(get_path("output_generated_plots"), "validation_dashboard.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\n[SUCCESS] Graph generated: {save_path}")
    print("Open this image to confirm biological realism.")
    plt.show()

if __name__ == "__main__":
    # Optional CLI arg: path to ID results directory
    if len(sys.argv) > 1:
        RESULTS_DIR = sys.argv[1]
    plot_dashboard()