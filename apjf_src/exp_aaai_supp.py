"""
AAAI 补充实验: LPIPS 图像质量 + 多次运行标准差
用法:
  /home/oyp/miniconda3/envs/apjf/bin/python -u apjf_src/exp_aaai_supp.py
"""
import sys, os, time, json
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms as T
import lpips

from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.joint_config import *

DEVICE = 'cuda'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

class SimpleImgDataset(Dataset):
    def __init__(self, image_dir, num=50):
        self.transform = T.Compose([T.Resize((256,256)), T.ToTensor()])
        paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])
        self.paths = paths[:num]
    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert('RGB'))
    def __len__(self):
        return len(self.paths)

# ── LPIPS ──
def calc_lpips():
    print('\n=== AA-3a: LPIPS ===')
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    ds = SimpleImgDataset(VAL_DIR, num=50)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    wm = WatermarkBranch(OMNIGUARD_CKPT, DEVICE).eval()
    for p in wm.parameters(): p.requires_grad = False
    vals = []
    for img in tqdm(loader, desc='LPIPS'):
        img = img.to(DEVICE)
        secret = torch.ones_like(img) * 0.5
        with torch.no_grad():
            stego, _, _ = wm.forward(img, secret, apply_noise=True)
        d = lpips_fn(img, stego).item()
        vals.append(d)
    avg = float(np.mean(vals))
    print(f'  LPIPS: {avg:.4f}')
    return avg

# ── 多次运行检测精度（用不同随机种子评估，非重训） ──
def eval_multi_seed():
    print('\n=== AA-3b: 3次不同随机种子 ===')
    from apjf_src.run_cvpr import load_models, EvalDataset, eval_batch
    from torch.utils.data import DataLoader

    seeds = [42, 123, 999]
    accs = []
    for seed in seeds:
        ds = EvalDataset(VAL_DIR, num_samples=200, seed=seed)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        wm, spn, fusion = load_models()
        correct, total = 0, 0
        for batch in tqdm(loader, desc=f'Seed {seed}'):
            logits, _, _, _, _ = eval_batch(wm, spn, fusion, batch)
            correct += (logits.argmax(dim=1) == batch['verdict'].to(DEVICE)).sum().item()
            total += len(batch['verdict'])
        acc = correct / total * 100
        accs.append(acc)
        print(f'  Seed {seed}: {acc:.2f}%')
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f'  Mean: {mean_acc:.2f}% ± {std_acc:.2f}%')
    return {'mean': float(mean_acc), 'std': float(std_acc), 'all': accs}

# ── 汇总 ──
results = {}
results['lpips'] = calc_lpips()
results['multi_seed'] = eval_multi_seed()

out = os.path.join(RESULTS_DIR, 'aaai_supp_results.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to {out}')
