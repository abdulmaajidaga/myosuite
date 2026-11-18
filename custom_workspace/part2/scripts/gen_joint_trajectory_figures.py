"""
Generate figures 3 and 4: group mean joint angle trajectories.

Fig 3 (joint_trajectories_original.png):
  Group mean trajectories under standard no-damping IK (output/originals/).
  Shows trunk is near-zero and undifferentiated between stroke and healthy.

Fig 4 (joint_trajectories_weighted.png):
  Group mean trajectories under SVD Two-Step IK (output/neutral_start/).
  Shows clear trunk compensation separation.

Output saved to paper_publication_assets/figures_and_tables/.

Run:
  python scripts/gen_joint_trajectory_figures.py
  python scripts/gen_joint_trajectory_figures.py --fig 3   # fig 3 only
  python scripts/gen_joint_trajectory_figures.py --fig 4   # fig 4 only
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_project_root

ROOT    = get_project_root()
OUT_DIR = os.path.join(ROOT, "paper_publication_assets", "figures_and_tables")

EXCLUDE_HEALTHY = {'01_12_1', '01_12_2', '01_12_3', '15_12_1'}

# Joints to plot — 2×3 grid
PLOT_JOINTS = [
    ('trunk_forward_lean', 'Trunk Forward Lean'),
    ('elv_angle',          'Elev Angle'),
    ('shoulder_elv',       'Shoulder Elevation'),
    ('shoulder_rot',       'Shoulder Rotation'),
    ('elbow_flexion',      'Elbow Flexion'),
    ('pro_sup',            'Pronation/Supination'),
]

N_FRAMES    = 100
RAD2DEG     = 57.2958
STROKE_COL  = '#d62728'
HEALTHY_COL = '#4472c4'


def load_mot(path):
    with open(path) as f:
        lines = f.readlines()
    hend = next(i for i, l in enumerate(lines) if 'endheader' in l)
    cols = lines[hend + 1].strip().split('\t')
    arr  = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()],
                    dtype=float)
    return {c: arr[:, i] for i, c in enumerate(cols)}


def normalise(arr, n=N_FRAMES):
    """Time-normalise a 1D trajectory to n frames."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n)
    return interp1d(x_old, arr, kind='linear')(x_new)


def load_group(run_dir, group):
    """Load all MOTs for a group, return dict of joint -> (n_subjects, N_FRAMES) array."""
    mot_dir = os.path.join(run_dir, group, 'mot')
    if not os.path.exists(mot_dir):
        return {}

    # Use batch_report if available and non-empty, otherwise use all MOT files
    rep_path = os.path.join(run_dir, 'batch_report.csv')
    ok_files = None
    if os.path.exists(rep_path):
        rep = pd.read_csv(rep_path)
        ok_files = set(rep[(rep['group'] == group) & (rep['status'] == 'OK')]['file'].tolist())
    if not ok_files:
        ok_files = {os.path.splitext(f)[0] for f in os.listdir(mot_dir) if f.endswith('.mot')}

    if group == 'healthy':
        ok_files -= EXCLUDE_HEALTHY

    trajs = {j: [] for j, _ in PLOT_JOINTS}
    for fid in sorted(ok_files):
        path = os.path.join(mot_dir, f'{fid}.mot')
        if not os.path.exists(path):
            continue
        mot = load_mot(path)
        for j, _ in PLOT_JOINTS:
            if j in mot:
                trajs[j].append(normalise(mot[j] * RAD2DEG))

    return {j: np.array(v) for j, v in trajs.items() if v}


def plot_trajectories(run_dir, title, out_path):
    stroke_data  = load_group(run_dir, 'stroke')
    healthy_data = load_group(run_dir, 'healthy')

    n_stroke  = len(next(iter(stroke_data.values()),  []))
    n_healthy = len(next(iter(healthy_data.values()), []))
    print(f"  {title}: {n_stroke} stroke, {n_healthy} healthy")

    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.sans-serif':  ['Arial', 'DejaVu Sans'],
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'axes.linewidth':   0.8,
        'xtick.major.width':0.8,
        'ytick.major.width':0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 5.5),
                             facecolor='white')
    axes = axes.flatten()
    x = np.linspace(0, 100, N_FRAMES)

    for idx, (joint, label) in enumerate(PLOT_JOINTS):
        ax = axes[idx]
        ax.set_facecolor('white')

        for data, color, gname in [
            (healthy_data, HEALTHY_COL, 'Healthy'),
            (stroke_data,  STROKE_COL,  'Stroke'),
        ]:
            if joint not in data or len(data[joint]) == 0:
                continue
            mat  = data[joint]
            mean = mat.mean(axis=0)
            sd   = mat.std(axis=0)
            ax.plot(x, mean, color=color, lw=1.6, label=gname, solid_capstyle='round')
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)

        ax.set_title(label, fontsize=8.5, fontweight='semibold', pad=5, color='#222222')
        ax.set_xlabel('% Motion', fontsize=7.5, color='#444444', labelpad=3)
        ax.set_ylabel('Degrees',  fontsize=7.5, color='#444444', labelpad=3)
        ax.tick_params(labelsize=7, colors='#444444', pad=2)
        ax.set_xlim(0, 100)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        ax.tick_params(color='#888888')

        if idx == 0:
            ax.legend(fontsize=7, loc='best', frameon=False,
                      handlelength=1.4, handletextpad=0.5,
                      labelcolor='#333333')

    plt.tight_layout(pad=0.8, h_pad=2.0, w_pad=1.5)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig', type=int, choices=[3, 4], default=None,
                        help='Generate only fig 3 or fig 4 (default: both)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    configs = {
        3: (
            os.path.join(ROOT, 'output', 'originals'),
            'Group Mean Joint Trajectories — Standard IK (No Damping, Free Trunk)',
            os.path.join(OUT_DIR, 'joint_trajectories_original.png'),
        ),
        4: (
            os.path.join(ROOT, 'output', 'neutral_start'),
            'Group Mean Joint Trajectories — SVD Two-Step Weighted IK',
            os.path.join(OUT_DIR, 'joint_trajectories_weighted.png'),
        ),
    }

    figs_to_run = [args.fig] if args.fig else [3, 4]

    for fig_num in figs_to_run:
        run_dir, title, out_path = configs[fig_num]
        print(f"\n=== Figure {fig_num} ===")
        if not os.path.exists(run_dir):
            print(f"  SKIP: {run_dir} does not exist")
            continue
        plot_trajectories(run_dir, title, out_path)


if __name__ == '__main__':
    main()
