import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use absolute paths for reliability
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(BASE_DIR, "test/models/cvae")
OUTPUT_DIR = os.path.join(BASE_DIR, "test/output/analysis")
FIGURES_DIR = os.path.join(BASE_DIR, "test/figures")

STAGES = [0, 1, 2, 3]
AUGMENTS = ["dtw", "linear", "smote"]

def extract_metrics():
    all_metrics = []
    for stage in STAGES:
        for aug in AUGMENTS:
            stage_dir = os.path.join(OUTPUT_DIR, f"stage{stage}_{aug}")
            if not os.path.exists(stage_dir):
                print(f"Skipping missing: {stage_dir}")
                continue
            
            m = {"stage": stage, "aug": aug, "pass_rate": 0.0, "cci_rho": 0.0}
            
            # Extract Pass Rate
            litval_path = os.path.join(stage_dir, "literature_validation.md")
            if os.path.exists(litval_path):
                with open(litval_path, "r") as f:
                    for line in f:
                        if "Overall:" in line:
                            match = re.search(r'\((\d+)%\)', line)
                            if match: m["pass_rate"] = float(match.group(1))
                            break
            
            # Extract CCI Rho
            stats_path = os.path.join(stage_dir, "publication_stats.md")
            if os.path.exists(stats_path):
                in_corr = False
                with open(stats_path, "r") as f:
                    for line in f:
                        if "## B. Correlation Analysis" in line: in_corr = True
                        if in_corr and "| CCI |" in line:
                            parts = [p.strip() for p in line.split('|')]
                            if len(parts) >= 3:
                                rho_str = parts[2]
                                try:
                                    m["cci_rho"] = float(rho_str)
                                except ValueError:
                                    m["cci_rho"] = 0.0
                                break
            all_metrics.append(m)
    return pd.DataFrame(all_metrics)

def plot_training_curves():
    fig, axes = plt.subplots(4, 3, figsize=(15, 20), sharex=True, sharey=True)
    for i, stage in enumerate(STAGES):
        for j, aug in enumerate(AUGMENTS):
            history_file = os.path.join(MODELS_DIR, f"cvae_stage{stage}_{aug}_history.csv")
            ax = axes[i, j]
            if os.path.exists(history_file):
                df = pd.read_csv(history_file)
                ax.plot(df['epoch'], df['train'], label='Train')
                ax.plot(df['epoch'], df['val'], label='Val')
                ax.set_title(f"Stage {stage} + {aug}")
                ax.grid(True, alpha=0.3)
                if i == 0 and j == 0:
                    ax.legend()
            else:
                ax.text(0.5, 0.5, "MISSING", ha='center', va='center')
    
    plt.suptitle("CVAE Training History (Stages 0-3)", fontsize=20)
    fig.text(0.5, 0.04, 'Epochs', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Loss (Reconstruction + KL)', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, "training_curves_all_stages.png"), dpi=300)
    print(f"Saved: training_curves_all_stages.png")

def plot_performance_evolution(df):
    if df.empty: return
    
    # Pass Rate Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='stage', y='pass_rate', hue='aug', marker='o', linewidth=2)
    plt.title("Literature Validation Pass Rate Evolution", fontsize=16)
    plt.ylabel("Pass Rate (%)", fontsize=12)
    plt.xlabel("Development Stage", fontsize=12)
    plt.xticks(STAGES)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "pass_rate_evolution.png"), dpi=300)
    print(f"Saved: pass_rate_evolution.png")
    
    # CCI Rho Plot (Clinical Validity)
    plt.figure(figsize=(10, 6))
    # CCI Rho should be highly negative for clinical validity (-0.9 is better than -0.5)
    # Let's plot absolute value or just raw rho
    sns.lineplot(data=df, x='stage', y='cci_rho', hue='aug', marker='s', linewidth=2)
    plt.axhline(-0.8, color='red', linestyle='--', label='Clinical Benchmark (Rho < -0.8)')
    plt.title("Clinical Validity Evolution (CCI vs. FMA Spearman Rho)", fontsize=16)
    plt.ylabel("Spearman Rho (Lower is Better Clinical Correlation)", fontsize=12)
    plt.xlabel("Development Stage", fontsize=12)
    plt.xticks(STAGES)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "clinical_validity_evolution.png"), dpi=300)
    print(f"Saved: clinical_validity_evolution.png")

def plot_trajectory_comparison():
    stage0_file = os.path.join(OUTPUT_DIR, "stage0_dtw/csv/FMA_20.csv")
    stage3_file = os.path.join(OUTPUT_DIR, "stage3_smote/csv/FMA_20.csv")
    
    if not (os.path.exists(stage0_file) and os.path.exists(stage3_file)):
        print("Trajectory comparison files missing!")
        return
        
    df0 = pd.read_csv(stage0_file)
    df3 = pd.read_csv(stage3_file)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 3D Plot (simulated in 2D for simplicity or just 2 planes)
    # Let's plot X-Y (Reaching plane)
    axes[0].plot(df0['Wr_x'], df0['Wr_y'], 'r--', label='Stage 0 (Baseline)')
    axes[0].plot(df3['Wr_x'], df3['Wr_y'], 'g-', linewidth=2, label='Stage 3 (Optimized)')
    axes[0].set_title("Wrist Trajectory Comparison (FMA 20 - X-Y Plane)", fontsize=14)
    axes[0].set_xlabel("X (Forward/Back)", fontsize=12)
    axes[0].set_ylabel("Y (Left/Right)", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Velocity Comparison
    v0 = np.sqrt(np.gradient(df0['Wr_x'])**2 + np.gradient(df0['Wr_y'])**2 + np.gradient(df0['Wr_z'])**2)
    v3 = np.sqrt(np.gradient(df3['Wr_x'])**2 + np.gradient(df3['Wr_y'])**2 + np.gradient(df3['Wr_z'])**2)
    
    axes[1].plot(v0, 'r--', label='Stage 0 Velocity')
    axes[1].plot(v3, 'g-', linewidth=2, label='Stage 3 Velocity')
    axes[1].set_title("Movement Velocity Profile (FMA 20)", fontsize=14)
    axes[1].set_xlabel("Time Step (normalized)", fontsize=12)
    axes[1].set_ylabel("Velocity (m/s equivalent)", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison_trajectories.png"), dpi=300)
    print(f"Saved: model_comparison_trajectories.png")

def main():
    df = extract_metrics()
    print(df)
    plot_training_curves()
    plot_performance_evolution(df)
    plot_trajectory_comparison()

if __name__ == "__main__":
    main()
