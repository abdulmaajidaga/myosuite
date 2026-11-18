import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(6.5, 3.2))

labels  = ['Wrist Dev.', 'Wrist Flex.', 'Pro/Sup', 'Elbow Flex.',
           'Shoulder Elv.', 'Shoulder Rot.', 'Elv. Angle',
           'SC r2', 'SC r3',
           'Trunk Fwd. Lean', 'Trunk Lat. Lean']
weights = [0.05]*7 + [0.20, 0.20] + [50.0, 50.0]
tiers   = [1]*7     + [2, 2]       + [3, 3]

colors = {1: '#4C9BE8', 2: '#F0A500', 3: '#E84C4C'}
bar_colors = [colors[t] for t in tiers]

x = np.arange(len(labels))
bars = ax.bar(x, weights, color=bar_colors, edgecolor='white', linewidth=0.6, zorder=3)

ax.set_yscale('log')
ax.set_ylim(0.02, 200)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
ax.set_ylabel('Damping weight $w_i$  (log scale)', fontsize=9)
ax.set_title('Per-DOF Damping Weights: CNS Force-Reserve Hierarchy', fontsize=10, fontweight='bold')
ax.yaxis.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
ax.set_axisbelow(True)

# Ratio annotations on top of bars
ratios = {0.05: '1×', 0.20: '4×', 50.0: '1000×'}
for bar, w in zip(bars, weights):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.6,
            ratios[w], ha='center', va='bottom', fontsize=7, fontweight='bold')

# Tier shading bands
ax.axvspan(-0.5, 6.5,  alpha=0.06, color='#4C9BE8', zorder=0)
ax.axvspan(6.5,  8.5,  alpha=0.06, color='#F0A500', zorder=0)
ax.axvspan(8.5, 10.5,  alpha=0.06, color='#E84C4C', zorder=0)

# Legend
patches = [
    mpatches.Patch(color='#4C9BE8', alpha=0.7, label='Tier 1 — Arm DOFs ($w=0.05$)'),
    mpatches.Patch(color='#F0A500', alpha=0.7, label='Tier 2 — SC Girdle ($w=0.20$, 4×)'),
    mpatches.Patch(color='#E84C4C', alpha=0.7, label='Tier 3 — Trunk ($w=50.0$, 1000×)'),
]
ax.legend(handles=patches, fontsize=8, loc='upper left',
          framealpha=0.9, edgecolor='#cccccc')

# Reference line at 0.05
ax.axhline(0.05, color='#4C9BE8', linestyle=':', linewidth=1.0, alpha=0.5)

plt.tight_layout()
import os as _os
_out = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     'paper_publication_assets', 'figures_and_tables', 'damping_weights.png')
plt.savefig(_out, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"Saved {_out}")
