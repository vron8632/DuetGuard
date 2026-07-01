"""
DuetGuard 多数据集评估脚本
一键在 AIGC / Columbia / CASIA v2 上评估检测精度 + 汇总对比表

用法:
  /home/oyp/miniconda3/envs/apjf/bin/python -u apjf_src/eval_all.py
"""
import sys, os, time, json
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.joint_config import *

DEVICE = 'cuda'
BATCH_SIZE = 4

# ── 加载模型 ──
def load_models():
    wm = WatermarkBranch(OMNIGUARD_CKPT, DEVICE).eval()
    for p in wm.parameters(): p.requires_grad = False
    spn = SPNExtractor(fp_dim=FP_DIM, base_ch=64, num_blocks=2, expand_ratio=4).to(DEVICE)
    spn.load_state_dict(torch.load(SAVE_DIR+'/spn_finetuned_best.pth', map_location='cpu', weights_only=True))
    spn.eval()
    fusion = FusionLayer(fp_dim=FP_DIM, act_feat_dim=ACT_FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    fusion.load_state_dict(torch.load(SAVE_DIR+'/fusion_best.pth', map_location='cpu', weights_only=True))
    fusion.eval()
    return wm, spn, fusion

# ── 通用评估函数 ──
@torch.no_grad()
def evaluate(wm, spn, fusion, loader, name=''):
    correct = 0
    total = 0
    conf = torch.zeros(2, 2)
    for batch in tqdm(loader, desc=name):
        img = batch['image'].to(DEVICE)
        label = batch['label'].to(DEVICE)
        secret = torch.ones_like(img) * 0.5

        stego, snoisy, rec = wm.forward(img, secret, apply_noise=True)
        spn_feat, noise_map = spn(snoisy)
        diff_img = (rec - secret).abs()
        active_feat = F.adaptive_avg_pool2d(diff_img, (8, 8)).reshape(img.size(0), -1)
        quality = (1.0 - diff_img.mean(dim=[1, 2, 3])).clamp(0, 1)
        pce = noise_map.std(dim=[1, 2, 3], keepdim=True)

        logits, _ = fusion(
            {'active_feat': active_feat, 'quality': quality},
            {'spn_feat': spn_feat, 'pce': pce},
        )
        pred = logits.argmax(dim=1)
        # 映射: class 0=可信(authentic), class 1/2=篡改(forgery)
        pred_bin = (pred > 0).long()
        label_bin = (label > 0).long()

        correct += (pred_bin == label_bin).sum().item()
        total += label.size(0)
        for i in range(label.size(0)):
            conf[label_bin[i], pred_bin[i]] += 1

    acc = correct / total * 100
    return {'accuracy': acc, 'samples': total, 'confusion': conf.tolist()}


# ═══════════════════════════════════════════
#  数据集定义
# ═══════════════════════════════════════════

transform_256 = T.Compose([T.Resize((256, 256)), T.ToTensor()])

class GenericDataset(Dataset):
    """通用二分类数据集：authentic_dir 和 forgery_dir 各取 num_per_class 张"""
    def __init__(self, authentic_dir, forgery_dir, num_per_class=None, transform=transform_256):
        self.transform = transform
        self.samples = []  # (path, label)
        for d, label in [(authentic_dir, 0), (forgery_dir, 1)]:
            if not os.path.isdir(d):
                continue
            files = sorted([os.path.join(d, f) for f in os.listdir(d)
                           if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
            if num_per_class:
                files = files[:num_per_class]
            for f in files:
                self.samples.append((f, label))
        print(f'  {os.path.basename(authentic_dir)} vs {os.path.basename(forgery_dir)}: {len(self.samples)} 张')

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return {'image': img, 'label': torch.tensor(label, dtype=torch.long)}

    def __len__(self):
        return len(self.samples)

class ColumbiaDataset(Dataset):
    """Columbia: Au-*=authentic, Sp-*=spliced"""
    def __init__(self, root, transform=transform_256):
        self.transform = transform
        self.samples = []
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            label = 0 if d.startswith('Au') else 1
            for f in sorted(os.listdir(full)):
                if f.lower().endswith('.bmp'):
                    self.samples.append((os.path.join(full, f), label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return {'image': img, 'label': torch.tensor(label, dtype=torch.long)}

    def __len__(self):
        return len(self.samples)

class CASIADataset(Dataset):
    """CASIA v2: Au/ = authentic, Tp/ = tampered"""
    def __init__(self, root, num_per_class=None, transform=transform_256):
        self.transform = transform
        self.samples = []
        for d, label in [('Au', 0), ('Tp', 1)]:
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            files = sorted([os.path.join(full, f) for f in os.listdir(full)
                           if f.lower().endswith('.jpg') and d + '_' in f])
            if num_per_class:
                files = files[:num_per_class]
            for f in files:
                self.samples.append((f, label))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert('RGB'))
        return {'image': img, 'label': torch.tensor(label, dtype=torch.long)}

    def __len__(self):
        return len(self.samples)


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════

def print_sep(t):
    print(f'\n{"="*60}\n  {t}\n{"="*60}')

def main():
    print('='*60)
    print('  DuetGuard 多数据集评估')
    print(f'  Device: {DEVICE}')
    print('='*60)

    wm, spn, fusion = load_models()
    all_results = {}

    # ── 1. AIGC 测试集 ──
    print_sep('数据集 1/3: AIGC Test')
    ds = GenericDataset(
        '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/aigc_test/authentic',
        '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/aigc_test/forgery',
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_results['AIGC'] = evaluate(wm, spn, fusion, loader, 'AIGC')

    # ── 2. Columbia ──
    print_sep('数据集 2/3: Columbia')
    columbia_root = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/columbia/ImSpliceDataset'
    ds = ColumbiaDataset(columbia_root)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_results['Columbia'] = evaluate(wm, spn, fusion, loader, 'Columbia')

    # ── 3. CASIA v2 ──
    print_sep('数据集 3/3: CASIA v2')
    casia_root = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/casia2/CASIA2.0_revised'
    ds = CASIADataset(casia_root)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_results['CASIAv2'] = evaluate(wm, spn, fusion, loader, 'CASIA v2')

    # ── 汇总表 ──
    print('\n' + '='*60)
    print('  📊 检测精度汇总')
    print('='*60)
    print(f'  {"数据集":20s} {"样本数":>8s} {"Accuracy":>10s} {"真→真":>7s} {"假→假":>7s}')
    print(f'  {"-"*54}')
    for name, res in all_results.items():
        c = res['confusion']
        print(f'  {name:20s} {res["samples"]:>8d} {res["accuracy"]:>8.1f}% '
              f'{c[0][0]:>5.0f}/{c[0][0]+c[0][1]:<3.0f} {c[1][1]:>5.0f}/{c[1][0]+c[1][1]:<3.0f}')

    # ── 对比 OmniGuard / EditGuard 论文 ──
    print('\n' + '='*60)
    print('  对比 OmniGuard (CVPR 25) / EditGuard (CVPR 24)')
    print('='*60)
    print(f'  {"方法":15s} {"Columbia":>10s} {"CASIA":>10s}')
    print(f'  {"-"*37}')
    print(f'  {"OmniGuard":15s} {"93.1%*":>10s} {"—":>10s}')
    print(f'  {"EditGuard":15s} {"—":>10s} {"97.9%*":>10s}')
    print(f'  {"DuetGuard(ours)":15s} {all_results["Columbia"]["accuracy"]:>8.1f}%  {all_results["CASIAv2"]["accuracy"]:>8.1f}%')
    print(f'  * 论文报告值，测试设置可能不同，仅供参考')

    # ── 保存 ──
    out = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/results/multi_dataset_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n结果已保存: {out}')
    print(f'完成!')

if __name__ == '__main__':
    main()
