#!/usr/bin/env python3
"""Figure 1: End-to-end pipeline overview diagram.

Outputs:
    figures/pipeline_overview.png  (300 DPI)
    figures/pipeline_overview.pdf  (vector)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.unicode_minus': False,
})

BLUE      = '#1F77B4'
ART_FACE  = '#EBEBEB'
ART_EDGE  = '#AAAAAA'
DARK_GREY = '#333333'
WHITE     = '#FFFFFF'
TOOL_CLR  = '#C8DCFA'    # light blue for tool text inside stage blocks

# ── Figure setup ──────────────────────────────────────────────────────────────
# 16 cm × 6.5 cm  →  6.30" × 2.56"
fig, ax = plt.subplots(figsize=(6.30, 2.56))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Layout constants (normalised coords 0-1) ──────────────────────────────────
Y_MID = 0.52          # vertical centre of all blocks
S_H   = 0.72          # stage block height
A_H   = 0.50          # artifact box height
S_W   = 0.138         # stage block width
A_W   = 0.095         # artifact box width

# Stage titles and tool lines
stages = [
    ('Data &\nAugmentation',     'SMOTE · DTW\nButterworth filter',  r'§3'),
    ('Generative\nModel',        'PyTorch · CVAE\n(FiLM + CFG), $d_z=32$',     r'§4'),
    ('Inverse\nKinematics',      'MuJoCo IK solver\nMyoArm (27 DOF)',           r'§5.1'),
    ('Inverse Dynamics\n& Valid.','Static opt. (L-BFGS-B)\nNMF · EMG valid.',  r'§5.2'),
]

artifacts = [
    'Raw MHH mocap\n21 markers @ 200 Hz\n77 recordings',
    'Augmented dataset\n$\\sim$58k samples\nFMA 16–66',
    'Generated trajectories\n$(100,\\ 15)$ features',
    'Joint angles\n7 DOFs · .mot files',
    'Torques (Nm)\nmuscle act. (32)\nCCI, VAF',
]

# Interleave: A0 → S0 → A1 → S1 → A2 → S2 → A3 → S3 → A4
# Compute x centres
GAP = 0.014           # gap between elements
x = 0.010
centers = []
widths  = []
kinds   = []          # 'art' or 'stg'

for i in range(4):
    # artifact
    centers.append(x + A_W / 2); widths.append(A_W); kinds.append('art')
    x += A_W + GAP
    # stage
    centers.append(x + S_W / 2); widths.append(S_W); kinds.append('stg')
    x += S_W + GAP

# final artifact
centers.append(x + A_W / 2); widths.append(A_W); kinds.append('art')

# ── Draw artifacts ─────────────────────────────────────────────────────────────
art_idx = 0
for ci, (cx, w, kind) in enumerate(zip(centers, widths, kinds)):
    if kind != 'art':
        continue
    bx = cx - A_W / 2
    by = Y_MID - A_H / 2
    box = FancyBboxPatch((bx, by), A_W, A_H,
                         boxstyle='round,pad=0.008',
                         facecolor=ART_FACE, edgecolor=ART_EDGE,
                         linewidth=0.7, zorder=2, clip_on=False)
    ax.add_patch(box)
    ax.text(cx, Y_MID, artifacts[art_idx],
            ha='center', va='center', fontsize=5.8,
            color='#222222', zorder=3, clip_on=False,
            multialignment='center')
    art_idx += 1

# ── Draw stage blocks ─────────────────────────────────────────────────────────
stg_idx = 0
for ci, (cx, w, kind) in enumerate(zip(centers, widths, kinds)):
    if kind != 'stg':
        continue
    bx = cx - S_W / 2
    by = Y_MID - S_H / 2
    rect = FancyBboxPatch((bx, by), S_W, S_H,
                          boxstyle='round,pad=0.006',
                          facecolor=BLUE, edgecolor='none',
                          zorder=2, clip_on=False)
    ax.add_patch(rect)

    title, tools, ref = stages[stg_idx]
    # Title (bold white)
    ax.text(cx, Y_MID + 0.11, title,
            ha='center', va='center', fontsize=6.8,
            color=WHITE, fontweight='bold', zorder=3,
            clip_on=False, multialignment='center')
    # Tool text (light blue)
    ax.text(cx, Y_MID - 0.14, tools,
            ha='center', va='center', fontsize=5.5,
            color=TOOL_CLR, zorder=3, clip_on=False,
            multialignment='center')
    # Section reference below block
    ax.text(cx, Y_MID - S_H / 2 - 0.06, ref,
            ha='center', va='top', fontsize=6.0,
            color=DARK_GREY, style='italic', zorder=3, clip_on=False)
    stg_idx += 1

# ── Draw arrows between adjacent elements ────────────────────────────────────
def right_edge(ci):
    return centers[ci] + widths[ci] / 2

def left_edge(ci):
    return centers[ci] - widths[ci] / 2

for ci in range(len(centers) - 1):
    x_start = right_edge(ci)
    x_end   = left_edge(ci + 1)
    ax.annotate('',
                xy=(x_end, Y_MID), xytext=(x_start, Y_MID),
                xycoords='data', textcoords='data',
                arrowprops=dict(
                    arrowstyle='->', color=DARK_GREY,
                    lw=0.9, mutation_scale=7
                ), zorder=4, clip_on=False)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout(pad=0.05)
fig.savefig(os.path.join(out_dir, 'pipeline_overview.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(out_dir, 'pipeline_overview.pdf'),
            bbox_inches='tight', facecolor='white')
print('Saved pipeline_overview.{png,pdf}')
plt.close(fig)
