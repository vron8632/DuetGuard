#!/bin/bash
# ===================================================================
# DuetGuard 一键补充实验脚本
# 运行所有 P0/P1 快速实验 + 输出 P2/P3 扩展实验的指导
# 所有结果自动保存到 results/ 目录
# 用法: bash run_supplement_experiments.sh
# ===================================================================
set -e

BASE_DIR="/media/oyp/数据/Projects/042_image_forensic/DuetGuard"
PYTHON="/home/oyp/miniconda3/envs/apjf/bin/python"
cd "$BASE_DIR"

# ── 结果保存 ──
RESULTS_DIR="$BASE_DIR/results"
FIGS_DIR="$RESULTS_DIR/figures"
TEXT_LOG="$RESULTS_DIR/supplement_results.txt"
JSON_LOG="$RESULTS_DIR/supplement_results.json"
mkdir -p "$FIGS_DIR"

# 清空旧日志
echo "{}" > "$JSON_LOG"
echo "" > "$TEXT_LOG"

# 日志函数: 同时写文件 + 打印到终端
log() {
    echo -e "$@" | tee -a "$TEXT_LOG"
}

log_json() {
    # $1 = json key path, $2 = json value (string)
    # 简易实现: 只处理最外层 key
    local key="$1"
    local val="$2"
    $PYTHON -c "
import json
with open('$JSON_LOG') as f:
    d = json.load(f)
d['$key'] = $val
with open('$JSON_LOG', 'w') as f:
    json.dump(d, f, indent=2)
"
}

# ═══════════════════════════════════════════
log "╔══════════════════════════════════════════════════════════╗"
log "║       DuetGuard 补充实验                                ║"
log "║       结果自动保存到:                                    ║"
log "║         $TEXT_LOG"
log "║         $JSON_LOG"
log "║         $FIGS_DIR/"
log "╚══════════════════════════════════════════════════════════╝"
log ""

# ── CUDA 检查 ──
log "=== [环境检查] ==="
$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" >> "$TEXT_LOG" 2>&1
$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
"
echo ""

# ══════════════════════════════════════════════════════════════
log "================================================================"
log "  P0: 快速实验"
log "================================================================"
log ""

# ── P0-1: 2000 样本 COCO 评估 + 95% CI ──
log "--- P0-1: 扩大 COCO 测试集到 2000 样本 + 95% CI ---"
log "┌──────────────────────────────────────────────────────────────┐"
cat > /tmp/exp_expanded_eval.py << 'PYEOF'
import sys, os, torch, math, json
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
from torch.utils.data import DataLoader
from apjf_src.run_cvpr import load_models, eval_batch, EvalDataset
from apjf_src.joint_config import *
from tqdm import tqdm

TEXT_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.txt'
JSON_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.json'

def log_out(s):
    print(s)
    with open(TEXT_LOG, 'a') as f:
        f.write(s + '\n')

wm, spn, fusion = load_models()
ds = EvalDataset(VAL_DIR, num_samples=1000, seed=42)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

correct = 0
total = 0
conf = torch.zeros(3, 3)
for batch in tqdm(loader, desc='P0-1: Expanded Eval (2000 samples)'):
    logits, _, _, _, _ = eval_batch(wm, spn, fusion, batch)
    pred = logits.argmax(dim=1)
    v = batch['verdict'].to('cuda')
    correct += (pred == v).sum().item()
    total += v.size(0)
    for i in range(len(v)):
        conf[v[i], pred[i]] += 1

acc = correct / total * 100
z = 1.96
p = correct / total
denom = 1 + z*z/total
center = (p + z*z/(2*total)) / denom
margin = z * math.sqrt((p*(1-p)/total + z*z/(4*total*total))) / denom
ci_lo = center - margin
ci_hi = center + margin

log_out(f'  测试样本: {total}')
log_out(f'  准确率: {acc:.2f}% ({correct}/{total})')
log_out(f'  95% Wilson CI: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]')
log_out(f'  置信区间宽度: ±{margin*100:.2f}%')
log_out(f'  混淆矩阵:')
for i, label in enumerate(['Authentic','Suspicious','Forged']):
    row = '  '.join(f'{conf[i,j]:6.0f}' for j in range(3))
    log_out(f'    {label:12s}: {row}')
log_out('')

# 写入 JSON
with open(JSON_LOG) as f:
    d = json.load(f)
d['P0-1_expanded_eval'] = {
    'samples': total,
    'accuracy_pct': round(acc, 2),
    'correct': int(correct),
    'ci_95_lo_pct': round(ci_lo*100, 2),
    'ci_95_hi_pct': round(ci_hi*100, 2),
    'margin_pct': round(margin*100, 2),
    'confusion_matrix': conf.tolist()
}
with open(JSON_LOG, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
$PYTHON /tmp/exp_expanded_eval.py 2>&1 | tee -a "$TEXT_LOG"
echo ""

# ── P0-2: 多 seed t-SNE + PCA ──
log "--- P0-2: 多 seed t-SNE + PCA 投影 ---"
log "┌──────────────────────────────────────────────────────────────┐"
cat > /tmp/exp_tsne_pca.py << 'PYEOF'
import sys, os, torch, numpy as np, json
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch.nn.functional as F
from torch.utils.data import DataLoader
from apjf_src.run_cvpr import load_models, eval_batch, EvalDataset
from apjf_src.joint_config import *
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGS_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/figures'
TEXT_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.txt'
JSON_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.json'
os.makedirs(FIGS_DIR, exist_ok=True)

def log_out(s):
    print(s)
    with open(TEXT_LOG, 'a') as f:
        f.write(s + '\n')

wm, spn, fusion = load_models()
ds = EvalDataset(VAL_DIR, num_samples=400, seed=42)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

all_feats, all_labels = [], []
with torch.no_grad():
    for batch in tqdm(loader, desc='P0-2: Collecting embeddings'):
        base = batch['image'].to('cuda')
        donor = batch['donor'].to('cuda')
        mask = batch['tamper_mask'].to('cuda')
        secret = batch['secret'].to('cuda')
        verdict = batch['verdict'].to('cuda')
        is_tampered = (verdict > 0).bool()
        stego, snoisy, rec = wm.forward(base, secret, apply_noise=True)
        final_img = stego.clone()
        if is_tampered.any():
            for i in range(len(is_tampered)):
                if is_tampered[i]:
                    m = mask[i:i+1]
                    final_img[i] = stego[i] * (1 - m) + donor[i] * m
        rec_final = wm.decode(final_img)
        diff_img = (rec_final - secret).abs()
        active_feat = F.adaptive_avg_pool2d(diff_img, (8, 8)).reshape(base.size(0), -1)
        quality = (1.0 - diff_img.mean(dim=[1,2,3])).clamp(0,1)
        spn_feat, noise_map = spn(final_img)
        pce = noise_map.std(dim=[1,2,3], keepdim=True)
        e = torch.cat([active_feat, quality.unsqueeze(1), spn_feat, pce], dim=1)
        all_feats.append(e.cpu())
        all_labels.append(verdict.cpu())

X = torch.cat(all_feats, dim=0).numpy()
y = torch.cat(all_labels, dim=0).numpy()
label_names = ['Authentic', 'Suspicious', 'Forged']
colors = {'Authentic': '#2ecc71', 'Suspicious': '#f39c12', 'Forged': '#e74c3c'}

# PCA
log_out('  Computing PCA...')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
explained = pca.explained_variance_ratio_
sil_pca = silhouette_score(X_pca, y)
log_out(f'  解释方差: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, total={explained.sum():.3f}')
log_out(f'  Silhouette score (PCA): {sil_pca:.4f}')

fig, ax = plt.subplots(figsize=(5, 4))
for i, name in enumerate(label_names):
    m = y == i
    ax.scatter(X_pca[m,0], X_pca[m,1], c=colors[name], label=name, alpha=0.5, s=8)
ax.legend(fontsize=10)
ax.set_xlabel(f'PC1 ({explained[0]:.1%})', fontsize=11)
ax.set_ylabel(f'PC2 ({explained[1]:.1%})', fontsize=11)
ax.set_title('PCA of DuetGuard Fusion Embeddings', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.text(0.02, 0.98, f'Silhouette={sil_pca:.3f}', transform=ax.transAxes, va='top', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'pca_embeddings.png'), dpi=200)
plt.close()
log_out('  ✅ results/figures/pca_embeddings.png')

# t-SNE 多 seed
tsne_results = {}
for seed in [0, 42, 99]:
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    sil_tsne = silhouette_score(X_tsne, y)
    tsne_results[f'seed_{seed}'] = round(float(sil_tsne), 4)
    log_out(f'  t-SNE seed={seed}: Silhouette={sil_tsne:.4f}')
    fig, ax = plt.subplots(figsize=(5,4))
    for i, name in enumerate(label_names):
        m = y == i
        ax.scatter(X_tsne[m,0], X_tsne[m,1], c=colors[name], label=name, alpha=0.5, s=8)
    ax.legend(fontsize=10)
    ax.set_xlabel('t-SNE Dim 1', fontsize=11); ax.set_ylabel('t-SNE Dim 2', fontsize=11)
    ax.set_title(f't-SNE (seed={seed})', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'tsne_seed{seed}.png'), dpi=200)
    plt.close()
    log_out(f'  ✅ results/figures/tsne_seed{seed}.png')
log_out('')

# JSON
with open(JSON_LOG) as f:
    d = json.load(f)
d['P0-2_embedding_analysis'] = {
    'pca_explained_variance': {
        'pc1': round(float(explained[0]), 4),
        'pc2': round(float(explained[1]), 4),
        'total': round(float(explained.sum()), 4),
    },
    'silhouette_pca': round(float(sil_pca), 4),
    'silhouette_tsne': tsne_results,
}
with open(JSON_LOG, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
$PYTHON /tmp/exp_tsne_pca.py 2>&1 | tee -a "$TEXT_LOG"
echo ""

# ══════════════════════════════════════════════════════════════
log "================================================================"
log "  P1: 推荐完成的实验"
log "================================================================"
log ""

# ── P1-1: 零样本评估 ──
log "--- P1-1: 零样本评估（AIGC / Columbia / CASIA v2）---"
log "┌──────────────────────────────────────────────────────────────┐"
$PYTHON -u apjf_src/eval_all.py 2>&1 | tee -a "$TEXT_LOG"

# 将 eval_all.py 的 JSON 结果合并到总 JSON 中
if [ -f "$RESULTS_DIR/multi_dataset_results.json" ]; then
    $PYTHON -c "
import json
with open('$RESULTS_DIR/supplement_results.json') as f:
    d = json.load(f)
with open('$RESULTS_DIR/multi_dataset_results.json') as f:
    zero = json.load(f)
d['P1-1_zero_shot'] = zero
with open('$RESULTS_DIR/supplement_results.json', 'w') as f:
    json.dump(d, f, indent=2)
print('  ✅ Zero-shot results merged into supplement_results.json')
"
fi
echo ""

# ── P1-2: 强攻击测试 ──
log "--- P1-2: 强攻击测试（JPEG q=15, Gaussian σ=0.1/0.2）---"
log "┌──────────────────────────────────────────────────────────────┐"
cat > /tmp/exp_strong_attacks.py << 'PYEOF'
import sys, os, io, torch, numpy as np, json
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from apjf_src.run_cvpr import load_models, eval_batch, EvalDataset, apply_jpeg
from apjf_src.joint_config import *

TEXT_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.txt'
JSON_LOG = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/supplement_results.json'

def log_out(s):
    print(s)
    with open(TEXT_LOG, 'a') as f:
        f.write(s + '\n')

wm, spn, fusion = load_models()
ds = EvalDataset(VAL_DIR, num_samples=100, seed=42)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

attacks = [
    ('Clean',         lambda x: x),
    ('JPEG q=70',     lambda x: apply_jpeg(x, 70)),
    ('JPEG q=50',     lambda x: apply_jpeg(x, 50)),
    ('JPEG q=30',     lambda x: apply_jpeg(x, 30)),
    ('JPEG q=15',     lambda x: apply_jpeg(x, 15)),
    ('Noise σ=0.01',  lambda x: (x + torch.randn_like(x)*0.01).clamp(0,1)),
    ('Noise σ=0.05',  lambda x: (x + torch.randn_like(x)*0.05).clamp(0,1)),
    ('Noise σ=0.10',  lambda x: (x + torch.randn_like(x)*0.10).clamp(0,1)),
    ('Noise σ=0.20',  lambda x: (x + torch.randn_like(x)*0.20).clamp(0,1)),
]

results = {}
header = f'  {"攻击类型":25s} {"准确率":>10s} {"正确/总数":>12s}'
log_out(header)
log_out('  ' + '-'*49)
for name, attack_fn in attacks:
    correct = 0; total = 0
    for batch in tqdm(loader, desc=name):
        base = batch['image'].cuda()
        donor = batch['donor'].cuda()
        mask = batch['tamper_mask'].cuda()
        secret = batch['secret'].cuda()
        verdict = batch['verdict'].cuda()
        is_tampered = (verdict > 0).bool()
        stego, snoisy, rec = wm.forward(base, secret, apply_noise=True)
        final_img = stego.clone()
        if is_tampered.any():
            for i in range(len(is_tampered)):
                if is_tampered[i]:
                    m = mask[i:i+1]
                    final_img[i] = stego[i] * (1 - m) + donor[i] * m
        final_img = attack_fn(final_img)
        rec_final = wm.decode(final_img)
        diff_img = (rec_final - secret).abs()
        active_feat = torch.nn.functional.adaptive_avg_pool2d(diff_img, (8,8)).reshape(base.size(0),-1)
        quality = (1.0 - diff_img.mean(dim=[1,2,3])).clamp(0,1)
        spn_feat, noise_map = spn(final_img)
        pce = noise_map.std(dim=[1,2,3], keepdim=True)
        logits, _ = fusion(
            {'active_feat': active_feat, 'quality': quality},
            {'spn_feat': spn_feat, 'pce': pce},
        )
        pred = logits.argmax(dim=1)
        correct += (pred == verdict).sum().item()
        total += verdict.size(0)
    acc = correct / total * 100
    results[name] = round(acc, 1)
    log_out(f'  {name:25s} {acc:>8.1f}%   {correct:>4d}/{total:<4d}')

# JSON
with open(JSON_LOG) as f:
    d = json.load(f)
d['P1-2_robustness_strong_attacks'] = results
with open(JSON_LOG, 'w') as f:
    json.dump(d, f, indent=2)
log_out('')
PYEOF
$PYTHON /tmp/exp_strong_attacks.py 2>&1 | tee -a "$TEXT_LOG"
echo ""

# ══════════════════════════════════════════════════════════════
log "================================================================"
log "  P2-P3: 长时间实验指导"
log "================================================================"
log ""
log "以下实验需重训练（数小时），请参考以下命令手动执行："
log ""
log "  [P2] SPN σ 消融:"
log "    # σ=0.001"
log "    sed -i 's/NOISE_STD = 0.01/NOISE_STD = 0.001/' apjf_src/spn_config.py"
log "    $PYTHON apjf_src/train_spn.py"
log "    cp weights/spn_extractor_best.pth weights/spn_extractor_sigma_0.001.pth"
log "    # σ=0.05"
log "    sed -i 's/NOISE_STD = 0.001/NOISE_STD = 0.05/' apjf_src/spn_config.py"
log "    $PYTHON apjf_src/train_spn.py"
log "    cp weights/spn_extractor_best.pth weights/spn_extractor_sigma_0.05.pth"
log ""
log "  [P3] Oracle baseline:"
log "    见 run_supplement_experiments.sh 末尾注释"
log ""

# ══════════════════════════════════════════════════════════════
log "================================================================"
log "  全部实验完成！"
log "================================================================"
log ""
log "结果文件位置:"
log "  📄 文本报告:  $TEXT_LOG"
log "  📊 结构化数据: $JSON_LOG"
log "  🖼️  图片:      $FIGS_DIR/"
log ""
log "快速查看全部结果:"
log "  cat $TEXT_LOG"
log ""

# ── 统计结果概览 ──
log "┌──────────────────────────────────────────────────────────────┐"
log "│  P0-1: COCO 2000 样本评估 + 95% CI                        │"
log "│  P0-2: t-SNE (seed=0,42,99) + PCA 嵌入分析                │"
log "│  P1-1: AIGC / Columbia / CASIA v2 零样本评估              │"
log "│  P1-2: 强攻击测试（JPEG q=15, σ=0.1/0.2）                │"
log "│                                                        │"
log "│  论文使用提示:                                            │"
log "│  PCA 图 → cp results/figures/pca_embeddings.png fig/    │"
log "│  然后在 4_experiments.tex 中引用                        │"
log "└──────────────────────────────────────────────────────────────┘"
