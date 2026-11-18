import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'DejaVu Sans'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
})

fig, ax = plt.subplots(figsize=(6, 3.0))

labels  = ['Wrist Dev.', 'Wrist Flex.', 'Pro/Sup', 'Elbow Flex.',
           'Shoulder Elv.', 'Shoulder Rot.', 'Elv. Angle',
           'SC r2', 'SC r3']
weights = [0.05]*7 + [0.20, 0.20]
tiers   = [1]*7    + [2, 2]

TIER1_COL = '#4C9BE8'
TIER2_COL = '#F0A500'
bar_colors = [TIER1_COL if t == 1 else TIER2_COL for t in tiers]

x = np.arange(len(labels))
bars = ax.bar(x, weights, color=bar_colors, edgecolor='white', linewidth=0.5,
              width=0.6, zorder=3)

ax.set_yscale('log')
ax.set_ylim(0.02, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=38, ha='right', fontsize=8, color='#333333')
ax.set_ylabel('Damping weight $w_i$  (log scale)', fontsize=8.5, color='#444444', labelpad=4)
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines['left'].set_color('#888888')
ax.spines['bottom'].set_color('#888888')
ax.tick_params(colors='#555555')

# Ratio labels above bars
ratios = {0.05: '1×', 0.20: '4×'}
for bar, w in zip(bars, weights):
    ax.text(bar.get_x() + bar.get_width()/2, w * 1.5,
            ratios[w], ha='center', va='bottom', fontsize=7,
            fontweight='semibold', color='#333333')

# Subtle tier shading
ax.axvspan(-0.5, 6.5, alpha=0.05, color=TIER1_COL, zorder=0)
ax.axvspan(6.5,  8.5, alpha=0.05, color=TIER2_COL, zorder=0)

# Legend
patches = [
    mpatches.Patch(color=TIER1_COL, alpha=0.8, label='Tier 1 — Arm DOFs ($w = 0.05$)'),
    mpatches.Patch(color=TIER2_COL, alpha=0.8, label='Tier 2 — SC Girdle ($w = 0.20$, 4×)'),
]
ax.legend(handles=patches, fontsize=7.5, loc='upper left',
          frameon=False, handlelength=1.2, handletextpad=0.5)

ax.axhline(0.05, color=TIER1_COL, linestyle=':', linewidth=0.9, alpha=0.4)

plt.tight_layout(pad=0.8)
import os as _os
_out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     'paper_publication_assets', 'figures_and_tables', 'damping_weights.png')
plt.savefig(_out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved {_out}")
