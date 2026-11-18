#!/usr/bin/env python3
"""Figure 2: CVAE architecture diagram (FiLM conditioning + CFG dropout).

Outputs:
    figures/cvae_architecture.png  (300 DPI)
    figures/cvae_architecture.pdf  (vector)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import matplotlib.patheffects as pe
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.unicode_minus': False,
})

# ── Colour palette ─────────────────────────────────────────────────────────────
ENC_WASH  = '#DFF0FF'   # encoder region background
LAT_WASH  = '#EFE0FF'   # latent region background
DEC_WASH  = '#DFFFDF'   # decoder region background
ENC_BLOCK = '#5599CC'   # encoder blocks
LAT_BLOCK = '#8860BB'   # latent block
DEC_BLOCK = '#3DAA5A'   # decoder blocks
FILM_CLR  = '#FF7F0E'   # FiLM block
WHITE     = '#FFFFFF'
DARK      = '#1A1A1A'
ARROW_CLR = '#333333'

# ── Figure: 16 cm × 8 cm → 6.30" × 3.15" ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.30, 3.15))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')
fig.patch.set_facecolor('white')


# ─── Helper: filled rounded box with centred text ─────────────────────────────
def draw_box(cx, cy, w, h, label, fcolor, tcolor=WHITE, fontsize=7.2,
             edgecolor='none', lw=1.0, bold=False, multialign='center'):
    bx, by = cx - w / 2, cy - h / 2
    box = FancyBboxPatch((bx, by), w, h,
                         boxstyle='round,pad=0.10',
                         facecolor=fcolor, edgecolor=edgecolor,
                         linewidth=lw, zorder=3, clip_on=False)
    ax.add_patch(box)
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fontsize, color=tcolor,
            fontweight='bold' if bold else 'normal',
            zorder=4, clip_on=False, multialignment=multialign)
    return bx, by, w, h


def arrow(x0, y0, x1, y1, dashed=False, color=ARROW_CLR, lw=1.1, ms=8):
    style = 'dashed' if dashed else 'solid'
    ax.annotate('',
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=ms,
                                linestyle=style),
                zorder=5, clip_on=False)


# ── Region shading ─────────────────────────────────────────────────────────────
# Encoder: x = 0.1 – 5.1
ax.add_patch(FancyBboxPatch((0.10, 0.30), 5.00, 7.40,
             boxstyle='round,pad=0.12', facecolor=ENC_WASH,
             edgecolor='#AACCEE', linewidth=0.7, zorder=1))
ax.text(2.60, 7.60, 'Encoder', ha='center', va='center',
        fontsize=7, color='#336699', fontstyle='italic')

# Latent: x = 5.4 – 8.4
ax.add_patch(FancyBboxPatch((5.40, 0.30), 3.00, 7.40,
             boxstyle='round,pad=0.12', facecolor=LAT_WASH,
             edgecolor='#AA88CC', linewidth=0.7, zorder=1))
ax.text(6.90, 7.60, 'Latent Space', ha='center', va='center',
        fontsize=7, color='#663399', fontstyle='italic')

# Decoder: x = 8.7 – 15.9
ax.add_patch(FancyBboxPatch((8.70, 0.30), 7.10, 7.40,
             boxstyle='round,pad=0.12', facecolor=DEC_WASH,
             edgecolor='#88BB88', linewidth=0.7, zorder=1))
ax.text(12.25, 7.60, 'Decoder', ha='center', va='center',
        fontsize=7, color='#226622', fontstyle='italic')

# ── ENCODER COLUMN (x ≈ 2.60) ─────────────────────────────────────────────────
EX = 2.60

# Input block
draw_box(EX, 6.30, 3.80, 0.85,
         r'Trajectory  $x \in \mathbb{R}^{100 \times 15}$',
         ENC_BLOCK, fontsize=6.6)
arrow(EX, 5.88, EX, 5.25)

# BiLSTM encoder
draw_box(EX, 4.72, 3.80, 0.95,
         'BiLSTM encoder\nhidden = 256',
         ENC_BLOCK, fontsize=6.6)

# Fork: two arrows from BiLSTM down to μ and logσ²
MU_X, LOGS_X = 1.50, 3.70
arrow(EX, 4.25, MU_X,   3.45)
arrow(EX, 4.25, LOGS_X, 3.45)

# μ block
draw_box(MU_X,   3.00, 1.60, 0.75,
         '$\\mu$\n$\\mathbb{R}^{32}$',
         ENC_BLOCK, fontsize=6.6)

# log σ² block
draw_box(LOGS_X, 3.00, 1.90, 0.75,
         '$\\log\\sigma^2$\n$\\mathbb{R}^{32}$',
         ENC_BLOCK, fontsize=6.6)

# ── LATENT COLUMN (x ≈ 6.90) ──────────────────────────────────────────────────
LX = 6.90

# Arrows from μ, logσ² into reparameterisation node
arrow(MU_X   + 0.80, 3.00, LX - 0.55, 3.00, color='#7755AA')
arrow(LOGS_X + 0.95, 3.00, LX - 0.55, 3.00, color='#7755AA')

# Reparameterization circle
theta = np.linspace(0, 2 * np.pi, 80)
r = 0.52
ax.fill(LX + r * np.cos(theta), 3.00 + r * np.sin(theta),
        color=LAT_BLOCK, zorder=3, clip_on=False)
ax.text(LX, 3.00,
        '$z = \\mu + \\sigma \\odot \\epsilon$\n$\\epsilon \\sim \\mathcal{N}(0, I)$',
        ha='center', va='center', fontsize=5.8, color=WHITE,
        zorder=4, clip_on=False, multialignment='center')

# ε label (incoming noise)
ax.text(LX, 4.20, '$\\epsilon \\sim \\mathcal{N}(0,I)$',
        ha='center', va='center', fontsize=6.0, color='#6633AA',
        zorder=4, clip_on=False)
arrow(LX, 3.95, LX, 3.52, color='#7755AA')

# z output label
ax.text(LX, 2.15, '$z \\in \\mathbb{R}^{32}$',
        ha='center', va='center', fontsize=6.5, color='#6633AA',
        fontweight='bold', zorder=4, clip_on=False)
arrow(LX, 2.48, LX, 2.35, color='#7755AA')

# Arrow from latent → decoder
arrow(LX + 0.55, 3.00, 9.70 - 1.90, 3.00, color='#7755AA')

# ── DECODER COLUMN (x ≈ 11.5) ─────────────────────────────────────────────────
DX = 11.50

# FMA score input (top)
draw_box(13.80, 6.60, 2.00, 0.70,
         'FMA score $c$',
         '#555555', fontsize=6.4)

# CFG dropout label
ax.text(13.80, 5.98,
        'dropout $p=0.10$ (CFG)',
        ha='center', va='top', fontsize=5.6,
        color='#AA4400', style='italic', zorder=4, clip_on=False)
arrow(13.80, 6.25, 13.80, 5.55, dashed=True, color='#AA4400', ms=7)

# FiLM block
draw_box(13.80, 5.05, 2.20, 0.85,
         'FiLM\nmodulation',
         FILM_CLR, fontsize=6.4)

# BiLSTM decoder
draw_box(DX, 4.00, 3.80, 1.70,
         'BiLSTM decoder\nhidden = 256',
         DEC_BLOCK, fontsize=6.6)

# FiLM → decoder arrows (three horizontal dashed arrows showing per-layer injection)
for dy in [4.40, 4.00, 3.60]:
    ax.annotate('',
                xy=(DX + 1.90, dy), xytext=(13.80 - 1.10, dy),
                arrowprops=dict(arrowstyle='->', color=FILM_CLR,
                                lw=0.9, mutation_scale=6,
                                linestyle='dashed'),
                zorder=5, clip_on=False)

# Arrow from decoder down to output
arrow(DX, 3.15, DX, 2.45)

# Output block
draw_box(DX, 2.00, 3.80, 0.80,
         r'Reconstructed $\hat{x} \in \mathbb{R}^{100 \times 15}$',
         DEC_BLOCK, fontsize=6.5)

# Loss equation below output block
loss_text = (r'$\mathcal{L} = \mathcal{L}_{\mathrm{pos}}'
             r' + 10\mathcal{L}_{\mathrm{vel}}'
             r' + 2.5\mathcal{L}_{\mathrm{acc}}'
             r' + 0.2\mathcal{L}_{\mathrm{KL}}'
             r' + 2\mathcal{L}_{\mathrm{dyn}}$')
ax.text(DX, 1.30, loss_text,
        ha='center', va='center', fontsize=6.0,
        color='#224422', zorder=4, clip_on=False)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
os.makedirs(out_dir, exist_ok=True)

plt.tight_layout(pad=0.05)
fig.savefig(os.path.join(out_dir, 'cvae_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(out_dir, 'cvae_architecture.pdf'),
            bbox_inches='tight', facecolor='white')
print('Saved cvae_architecture.{png,pdf}')
plt.close(fig)
