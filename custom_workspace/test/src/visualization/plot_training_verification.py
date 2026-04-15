"""
CVAE FMA Training Verification Dashboard

Comprehensive 11-panel verification dashboard comparing:
  - Row 1: Time-normalized trajectories (Wrist Y, Wrist Z, Trunk, Velocity)
  - Row 2: Box plots of kinematic metrics (Wrist Range, Peak Velocity, Trunk Disp, Jerk)
  - Row 3: Latent space PCA, FMA vs Motion Range, FMA vs Trunk Comp, Model Info card
"""
import os
import sys
import glob
import warnings
import re
import argparse

import numpy as np
import pandas as pd
import torch
import joblib
from scipy import stats
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.utils.config import get_path, get

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
ALL_COLS = ARM_COLS + TRUNK_COLS

# Color scheme
C_HEALTHY = '#2ca02c'
C_STROKE = '#d62728'

# =============================================================================
# CLI Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model-version", type=str, default="v2", 
                    choices=["v1", "v2", "v3", "stage0", "stage1", "stage2", "stage3"])
parser.add_argument("--data-source", type=str, default="smote", choices=["dtw", "smote", "linear"])
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--gen-dir", type=str, default=None)
args_cli = parser.parse_args()

# =============================================================================
# Paths
# =============================================================================
# Dynamic model import
if args_cli.model_version.startswith("stage"):
    stage_num = int(args_cli.model_version.replace("stage", ""))
    from src.generation.minimalist_models import STAGE_MODELS
    MotionCVAE = STAGE_MODELS[stage_num]
    MODEL_PATH = os.path.join(BASE_DIR, f"models/cvae/cvae_{args_cli.model_version}_{args_cli.data_source}_best.pth")
else:
    from src.generation.model_v2 import MotionCVAE # Default
    MODEL_PATH = os.path.join(BASE_DIR, f"models/cvae/cvae_cutoff_fma_{args_cli.model_version}_{args_cli.data_source}_best.pth")

# Data paths
PROC_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/processed")
if args_cli.data_source == "smote":
    AUG_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented_smote")
elif args_cli.data_source == "linear":
    AUG_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented_backup")
    if not os.path.exists(AUG_DIR): AUG_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented_linear")
else:
    AUG_DIR = os.path.join(BASE_DIR, "data/kinematic/cutoff/augmented")

GEN_DIR = args_cli.gen_dir if args_cli.gen_dir else os.path.join(BASE_DIR, "output/generated/csv")
SCORES_PATH = os.path.join(BASE_DIR, "output/scores.csv")
SCALER_PATH = os.path.join(BASE_DIR, "models/cvae/scaler_cutoff_fma.pkl")
OUTPUT_DIR = args_cli.output_dir if args_cli.output_dir else os.path.join(BASE_DIR, "output/analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Data helpers
# =============================================================================
def to_delta(data):
    return data - data[0:1]

def load_scores():
    if not os.path.exists(SCORES_PATH): return {}
    df = pd.read_csv(SCORES_PATH)
    mapping = {}
    for _, row in df.iterrows():
        stem = row['filename'].replace('.mot', '')
        mapping[stem] = int(row['fma_score'])
    return mapping

def load_real_data(scores_map):
    healthy_data, stroke_data = [], []
    healthy_scores, stroke_scores = [], []
    if not os.path.exists(PROC_DIR): return np.array([]), [], np.array([]), []
    for f in sorted(glob.glob(os.path.join(PROC_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        for c in ALL_COLS:
            if c not in df.columns: df[c] = 0.0
        data = to_delta(df[ALL_COLS].values)
        if stem.startswith('S'):
            stroke_data.append(data)
            stroke_scores.append(scores_map.get(stem, 19))
        else:
            healthy_data.append(data)
            healthy_scores.append(66)
    return (np.array(healthy_data), healthy_scores, np.array(stroke_data), stroke_scores)

def load_generated_data():
    gen_data = {}
    if not os.path.exists(GEN_DIR): return {}
    for f in sorted(glob.glob(os.path.join(GEN_DIR, "FMA_*.csv"))):
        bn = os.path.basename(f)
        try:
            fma = int(re.search(r'FMA_(\d+)', bn).group(1))
            df = pd.read_csv(f)
            for c in ALL_COLS:
                if c not in df.columns: df[c] = 0.0
            gen_data[fma] = to_delta(df[ALL_COLS].values)
        except: continue
    return gen_data

def _gen_colors(fma_scores):
    cmap = plt.cm.RdYlGn
    scores = sorted(fma_scores)
    if not scores: return {}, {}
    lo, hi = min(scores), max(scores)
    span = max(1, hi - lo)
    colors = {s: cmap((s - lo) / span) for s in scores}
    labels = {s: f'Gen FMA {s}' for s in scores}
    return colors, labels

def load_augmented_sample(n_per_bin=50):
    if not os.path.exists(AUG_DIR): return {}
    all_files = glob.glob(os.path.join(AUG_DIR, "*.csv"))
    fma_files = {}
    for f in all_files:
        bn = os.path.basename(f)
        match = re.search(r'FMA(\d+)', bn)
        if match:
            fma = int(match.group(1))
            fma_files.setdefault(fma, []).append(f)
    
    aug_sampled = {}
    for fma, files in fma_files.items():
        sample = np.random.choice(files, min(len(files), n_per_bin), replace=False)
        data_list = []
        for f in sample:
            df = pd.read_csv(f)
            for c in ALL_COLS:
                if c not in df.columns: df[c] = 0.0
            data_list.append(df[ALL_COLS].values) # Augmented is already delta
        aug_sampled[fma] = np.array(data_list)
    return aug_sampled

def compute_metrics(data):
    wrist_y = data[:, 7]
    peak_vel = np.max(np.sqrt(np.sum(np.gradient(data[:, 6:9], axis=0)**2, axis=1)))
    trunk_disp = np.max(np.sqrt(np.sum(data[:, 12:15]**2, axis=1)))
    acc = np.gradient(np.gradient(data[:, 6:9], axis=0), axis=0)
    jerk = np.sqrt(np.sum(np.gradient(acc, axis=0)**2, axis=1))
    return {
        'wrist_range_y': np.max(wrist_y) - np.min(wrist_y),
        'peak_velocity': peak_vel,
        'trunk_disp': trunk_disp,
        'jerk_sum': np.sum(jerk)
    }

def plot_timeseries(ax, title, ylabel, healthy_data, stroke_data, gen_data, col_idx):
    colors, labels = _gen_colors(gen_data.keys())
    for fma in sorted(gen_data.keys()):
        ax.plot(gen_data[fma][:, col_idx], color=colors[fma], alpha=0.8, linewidth=1.5)
    
    h_curves = healthy_data[:, :, col_idx] if healthy_data.size > 0 else np.array([])
    s_curves = stroke_data[:, :, col_idx] if stroke_data.size > 0 else np.array([])
    
    if h_curves.size > 0:
        n = min(h_curves.shape[1], 100)
        ax.fill_between(range(n), np.percentile(h_curves, 25, axis=0), 
                        np.percentile(h_curves, 75, axis=0), color=C_HEALTHY, alpha=0.2)
        ax.plot(np.median(h_curves, axis=0), color=C_HEALTHY, linewidth=2, label='Real Healthy')
    
    if s_curves.size > 0:
        n = min(s_curves.shape[1], 100)
        ax.fill_between(range(n), np.percentile(s_curves, 25, axis=0), 
                        np.percentile(s_curves, 75, axis=0), color=C_STROKE, alpha=0.2)
        ax.plot(np.median(s_curves, axis=0), color=C_STROKE, linewidth=2, label='Real Stroke')
        
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)

def make_dashboard(healthy_data, healthy_scores, stroke_data, stroke_scores, gen_data, aug_sampled):
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    ax_wy = fig.add_subplot(gs[0, 0]); plot_timeseries(ax_wy, 'Wrist Y (Reach)', 'Pos (mm)', healthy_data, stroke_data, gen_data, 7)
    ax_wz = fig.add_subplot(gs[0, 1]); plot_timeseries(ax_wz, 'Wrist Z (Forward)', 'Pos (mm)', healthy_data, stroke_data, gen_data, 8)
    ax_tr = fig.add_subplot(gs[0, 2]); plot_timeseries(ax_tr, 'Trunk Y (Comp)', 'Pos (mm)', healthy_data, stroke_data, gen_data, 13)
    
    ax_vel = fig.add_subplot(gs[0, 3])
    colors, _ = _gen_colors(gen_data.keys())
    for fma in sorted(gen_data.keys()):
        v = np.sqrt(np.sum(np.gradient(gen_data[fma][:, 6:9], axis=0)**2, axis=1))
        ax_vel.plot(v, color=colors[fma], alpha=0.8)
    ax_vel.set_title('Wrist Velocity Profile', fontsize=10, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)

    # Boxplots
    metrics_h = [compute_metrics(d) for d in healthy_data] if healthy_data.size > 0 else []
    metrics_s = [compute_metrics(d) for d in stroke_data] if stroke_data.size > 0 else []
    metrics_g = {f: compute_metrics(d) for f, d in gen_data.items()}
    
    def plot_box(ax, key, title, unit):
        data = []
        if metrics_h: data.append([m[key] for m in metrics_h])
        if metrics_s: data.append([m[key] for m in metrics_s])
        labels = []
        if metrics_h: labels.append('Healthy')
        if metrics_s: labels.append('Stroke')
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], [C_HEALTHY, C_STROKE]): patch.set_facecolor(color); patch.set_alpha(0.5)
        
        for fma in sorted(metrics_g.keys()):
            ax.scatter(len(labels) + 0.5, metrics_g[fma][key], color=colors[fma], zorder=5, label=f'Gen {fma}')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(unit, fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_b1 = fig.add_subplot(gs[1, 0]); plot_box(ax_b1, 'wrist_range_y', 'Wrist Range Y', 'mm')
    ax_b2 = fig.add_subplot(gs[1, 1]); plot_box(ax_b2, 'peak_velocity', 'Peak Velocity', 'mm/f')
    ax_b3 = fig.add_subplot(gs[1, 2]); plot_box(ax_b3, 'trunk_disp', 'Trunk Comp', 'mm')
    ax_b4 = fig.add_subplot(gs[1, 3]); plot_box(ax_b4, 'jerk_sum', 'Cumulative Jerk', 'mm/f3')

    # PCA
    ax_pca = fig.add_subplot(gs[2, 0])
    pca = PCA(n_components=2)
    flat_h = healthy_data.reshape(healthy_data.shape[0], -1) if healthy_data.size > 0 else np.array([])
    if flat_h.size > 0:
        pca.fit(flat_h)
        h_pca = pca.transform(flat_h)
        ax_pca.scatter(h_pca[:, 0], h_pca[:, 1], c=C_HEALTHY, label='Healthy', alpha=0.5)
        for fma in sorted(gen_data.keys()):
            g_pca = pca.transform(gen_data[fma].reshape(1, -1))
            ax_pca.scatter(g_pca[:, 0], g_pca[:, 1], color=colors[fma], marker='X', s=100, edgecolors='black')
    ax_pca.set_title('Kinematic Latent Space (PCA)', fontsize=10, fontweight='bold')

    # FMA vs Metrics
    ax_fma_r = fig.add_subplot(gs[2, 1])
    aug_fmas = sorted(aug_sampled.keys())
    aug_ranges = [np.mean([compute_metrics(d)['wrist_range_y'] for d in aug_sampled[f]]) for f in aug_fmas]
    ax_fma_r.plot(aug_fmas, aug_ranges, 'o-', color='gray', alpha=0.3, label='Augmented')
    for fma in sorted(metrics_g.keys()):
        ax_fma_r.scatter(fma, metrics_g[fma]['wrist_range_y'], color=colors[fma], s=80, edgecolors='black', zorder=10)
    ax_fma_r.set_title('FMA vs Wrist Range', fontsize=10)

    ax_fma_t = fig.add_subplot(gs[2, 2])
    aug_trunks = [np.mean([compute_metrics(d)['trunk_disp'] for d in aug_sampled[f]]) for f in aug_fmas]
    ax_fma_t.plot(aug_fmas, aug_trunks, 's-', color='gray', alpha=0.3)
    for fma in sorted(metrics_g.keys()):
        ax_fma_t.scatter(fma, metrics_g[fma]['trunk_disp'], color=colors[fma], s=80, edgecolors='black', zorder=10)
    ax_fma_t.set_title('FMA vs Trunk Comp', fontsize=10)

    # Info
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis('off')
    info_text = f"FMA MODEL VERIFICATION\nStage: {args_cli.model_version}\nSource: {args_cli.data_source}\n\n"
    for fma in sorted(metrics_g.keys()):
        info_text += f"FMA {fma}: Range {metrics_g[fma]['wrist_range_y']:.1f}mm, Trunk {metrics_g[fma]['trunk_disp']:.1f}mm\n"
    ax_info.text(0, 1, info_text, transform=ax_info.transAxes, fontsize=9, family='monospace', va='top', bbox=dict(facecolor='lightyellow', alpha=0.5))

    out_path = os.path.join(OUTPUT_DIR, f'verification_{args_cli.model_version}_{args_cli.data_source}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    scores_map = load_scores()
    h_data, h_scores, s_data, s_scores = load_real_data(scores_map)
    gen_data = load_generated_data()
    aug_sampled = load_augmented_sample()
    make_dashboard(h_data, h_scores, s_data, s_scores, gen_data, aug_sampled)
