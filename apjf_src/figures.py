"""
DuetGuard 论文数据图生成脚本
生成: 消融柱状图, 鲁棒性折线图, 混淆矩阵热力图
用法: python apjf_src/figures.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

FIGS_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/cvpr-latex模版/fig'
os.makedirs(FIGS_DIR, exist_ok=True)

# 颜色方案 (CVPR 风格)
TEAL = '#028090'
NAVY = '#1e2761'
ORANGE = '#e67e22'
GREEN = '#2e7d32'
GRAY = '#999999'
LIGHT_GRAY = '#e0e0e0'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Fig 3: 消融柱状图 ──
def fig_ablation():
    labels = ['Watermark\nOnly (P0)', 'SPN\nNo Finetune', 'DuetGuard\n(P3 Full)']
    values = [50.0, 50.6, 99.4]
    colors = [GRAY, GRAY, TEAL]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor='white', linewidth=0.5)

    # 添加数据标签
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 随机线
    ax.axhline(y=50.0, color=ORANGE, linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(2.3, 51, 'Random (3-class)', color=ORANGE, fontsize=9, ha='right')

    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_ylim(0, 108)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'ablation.png'))
    plt.close(fig)
    print('  ✅ ablation.png')


# ── Fig 4: 鲁棒性折线图 ──
def fig_robustness():
    attacks = ['Clean', 'JPEG\nq=70', 'JPEG\nq=50', 'JPEG\nq=30', 'Noise\nσ=0.01', 'Noise\nσ=0.05']
    values = [99.0, 94.8, 90.8, 85.2, 98.5, 96.2]
    x = np.arange(len(attacks))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    # 填充区域
    ax.fill_between(x, values, 50, alpha=0.08, color=TEAL)

    # 折线 + 点
    ax.plot(x, values, '-o', color=TEAL, linewidth=2.2, markersize=8,
            markerfacecolor=TEAL, markeredgecolor='white', markeredgewidth=1.5)

    # 数据标签
    for xi, vi in zip(x, values):
        ax.annotate(f'{vi:.1f}%', (xi, vi), textcoords='offset points',
                    xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel('Detection Accuracy (%)')
    ax.set_ylim(50, 105)
    ax.axhline(y=50, color=ORANGE, linestyle='--', linewidth=1, alpha=0.5)
    ax.text(5.2, 51, 'Random', color=ORANGE, fontsize=8, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'robustness.png'))
    plt.close(fig)
    print('  ✅ robustness.png')


# ── Fig 6: 混淆矩阵热力图 ──
def fig_confusion():
    cm = np.array([[399, 0, 1],
                   [0, 0, 0],
                   [5, 0, 395]])
    classes = ['Authentic', 'Suspicious', 'Forgery']

    fig, ax = plt.subplots(figsize=(4, 3.5))

    # 自定义颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('teal_cm', ['#f0f9ff', '#028090'], N=256)

    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=400)

    # 在每个格子里写数字
    thresh = 200
    for i in range(3):
        for j in range(3):
            val = int(cm[i, j])
            color = 'white' if cm[i, j] > thresh else '#1a1a1a'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=16 if val > 0 else 10,
                    fontweight='bold' if val > 0 else 'normal',
                    color=color)

    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    ax.tick_params(axis='both', length=0)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, 'confusion.png'))
    plt.close(fig)
    print('  ✅ confusion.png')


if __name__ == '__main__':
    print('生成论文数据图...')
    fig_ablation()
    fig_robustness()
    fig_confusion()
    print(f'全部完成 → {FIGS_DIR}/')
