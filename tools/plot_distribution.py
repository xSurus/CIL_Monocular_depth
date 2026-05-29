import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
matplotlib.rcParams.update({'font.size': 9, 'font.family': 'serif'})

CLIP_DIR = '/content/data/clip_scores'

train = pd.read_csv(f'{CLIP_DIR}/train_categories.csv')
test  = pd.read_csv(f'{CLIP_DIR}/test_categories.csv')

cats   = ['aerial', 'water', 'vegetation', 'other']
labels = ['Aerial', 'Water', 'Vegetation', 'Other']
colors      = ['white',   '#4a90d9', '#5aaa6e', '#aaaaaa']
edge_colors = ['#888888', '#4a90d9', '#5aaa6e', '#aaaaaa']

t_pct = [100 * (train['category'] == c).mean() for c in cats]
v_pct = [100 * (test['category']  == c).mean() for c in cats]

x   = range(len(cats))
w   = 0.35
fig, ax = plt.subplots(figsize=(3.3, 1.3))

bars1 = ax.bar([i - w/2 for i in x], t_pct, w, color=colors, alpha=0.85,
               edgecolor=edge_colors, linewidth=0.8)
bars2 = ax.bar([i + w/2 for i in x], v_pct, w, color=colors, alpha=0.45,
               edgecolor=edge_colors, linewidth=1.2)

ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel('Fraction (%)')
ax.set_ylim(0, 95)
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

train_handle = mpatches.Patch(facecolor='#555555', alpha=0.85, label='Train')
test_handle  = mpatches.Patch(facecolor='#555555', alpha=0.45, edgecolor='#555555', linewidth=1.2, label='Test')
ax.legend(handles=[train_handle, test_handle], framealpha=0.9, fontsize=8)

def fmt(v): return '<1%' if 0 < v < 1 else f'{v:.0f}%'
for bar, val in zip(bars1, t_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            fmt(val), ha='center', va='bottom', fontsize=7)
for bar, val in zip(bars2, v_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            fmt(val), ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('reports/fig_distribution.pdf', bbox_inches='tight')
plt.savefig('reports/fig_distribution.png', dpi=200, bbox_inches='tight')
print('Saved reports/fig_distribution.pdf and .png')
