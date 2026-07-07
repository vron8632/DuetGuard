"""
DuetGuard 论文数据图 — 精修版 (seaborn 风格, 紧凑高级)
用法: python apjf_src/figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
import numpy as np
import os

FIGS_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/aaai-duetguard/fig'
os.makedirs(FIGS_DIR, exist_ok=True)

# ─── seaborn-v0_8 风格 ───
try:
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.0)
except ImportError:
    pass

# 配色
TEAL   = '#0d7c7c'
NAVY   = '#1a2a40'
ORANGE = '#e07730'
GREEN  = '#2e8b57'
GRAY   = '#a0a0a0'
LGRAY  = '#e8e8e8'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.4,
})


# ── Fig 3: 消融柱状图 ──
def fig_ablation():
    labels = ['Watermark\nOnly (P0)', 'SPN\nNo Finetune', 'DuetGuard\n(P3 Full)']
    values = [50.0, 50.6, 99.4]
    colors = [LGRAY, LGRAY, TEAL]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    bars = ax.bar(labels, values, color=colors, width=0.52, edgecolor='white', linewidth=0.4, zorder=3)

    for bar, v in zip(bars, values):
        c = TEAL if v > 90 else '#555'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.2,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color=c)

    # 随机基线
    ax.axhline(y=50.0, color=ORANGE, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.75)
    ax.text(2.35, 52, 'Random\n(3-class)', fontsize=7, ha='right', va='bottom', linespacing=1.1,
            color='#1a2a40',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_ylim(0, 112)
    ax.tick_params(axis='both', length=0)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS_DIR, 'ablation.png'))
    plt.close(fig)
    print('  ✅ ablation.png')


# ── Fig 4: 鲁棒性折线图 ──
def fig_robustness():
    attacks = ['Clean','JPEG 70','JPEG 50','JPEG 30','Noise\n0.01','Noise\n0.05']
    values  = [99.0, 94.8, 90.8, 85.2, 98.5, 96.2]
    x = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=(4.8, 2.6))

    # 填充区域
    ax.fill_between(x, values, 50, alpha=0.06, color=TEAL)

    # 折线 + 圆点
    ax.plot(x, values, '-o', color=TEAL, linewidth=1.8, markersize=6,
            markerfacecolor='white', markeredgecolor=TEAL, markeredgewidth=1.5, zorder=4)

    # 数据标签
    for xi, vi in zip(x, values):
        yoff =  10 if xi < 3 else 10
        ax.annotate(f'{vi:.1f}%', (xi, vi), textcoords='offset points',
                    xytext=(0, yoff), ha='center', fontsize=8, fontweight='bold', color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks, fontsize=7.5)
    ax.set_ylabel('Accuracy (%)', fontsize=8)
    ax.set_ylim(50, 106)
    ax.axhline(y=50, color=ORANGE, linestyle=(0, (4, 3)), linewidth=0.8, alpha=0.5)
    ax.text(5.1, 51.8, 'Random', fontsize=7, ha='right',
            color='#1a2a40',
            path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.tick_params(axis='both', length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS_DIR, 'robustness.png'))
    plt.close(fig)
    print('  ✅ robustness.png')


# ── Fig 5: 混淆矩阵热力图 ──
def fig_confusion():
    cm = np.array([[399, 0, 1],
                   [0, 0, 0],
                   [5, 0, 395]])
    classes = ['Authentic', 'Suspicious', 'Forgery']
    cm_normalized = cm / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('cmap', ['#f4fafb', '#0d7c7c'], N=256)
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=400, aspect='auto')

    thresh = 200
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            pct = cm_normalized[i, j]
            color = 'white' if cm[i, j] > thresh else '#222'
            label = f'{val}\n({pct:.1f}%)' if val > 0 else '0'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=8.5 if val > 0 else 7,
                    fontweight='bold' if val > 0 else 'normal',
                    color=color, linespacing=1.1)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8.5)
    ax.set_ylabel('Ground Truth', fontsize=8.5)
    ax.tick_params(axis='both', length=0)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS_DIR, 'confusion.png'))
    plt.close(fig)
    print('  ✅ confusion.png')


if __name__ == '__main__':
    print('生成论文数据图 (精修版)...')
    fig_ablation()
    fig_robustness()
    fig_confusion()
    print(f'全部完成 → {FIGS_DIR}/')
