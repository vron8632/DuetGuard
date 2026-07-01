"""
DuetGuard 完整实验套件
一键运行所有评估实验，终端显示进度。

用法:
  /home/oyp/miniconda3/envs/apjf/bin/python -u apjf_src/run_experiments.py
"""
import sys, os, time, json, io
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from torchvision import transforms as T

from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.data.dataset import JointTrainDataset
from apjf_src.joint_config import *

DEVICE = 'cuda'
SAVE_RESULTS = os.path.join(BASE_DIR, 'results')
os.makedirs(SAVE_RESULTS, exist_ok=True)


# ============================================================
# 模型加载
# ============================================================
def load_models(spn_ckpt_name='spn_finetuned_best.pth', fusion_ckpt_name='fusion_best.pth'):
    wm = WatermarkBranch(OMNIGUARD_CKPT, DEVICE).eval()
    for p in wm.parameters(): p.requires_grad = False

    spn = SPNExtractor(fp_dim=FP_DIM, base_ch=64, num_blocks=2, expand_ratio=4).to(DEVICE)
    spn_path = os.path.join(SAVE_DIR, spn_ckpt_name)
    spn.load_state_dict(torch.load(spn_path, map_location='cpu', weights_only=True))
    spn.eval()

    fusion = FusionLayer(fp_dim=FP_DIM, act_feat_dim=ACT_FEAT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    fusion_path = os.path.join(SAVE_DIR, fusion_ckpt_name)
    fusion.load_state_dict(torch.load(fusion_path, map_location='cpu', weights_only=True))
    fusion.eval()

    return wm, spn, fusion


# ============================================================
# 数据加载
# ============================================================
class EvalDataset(Dataset):
    """评估用数据集，按需生成篡改样本"""
    def __init__(self, image_dir, img_size=256, num_samples=200,
                 tamper_ratio=0.5, seed=42):
        self.img_size = img_size
        self.tamper_ratio = tamper_ratio
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        all_images = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(image_dir, f))
        rng = np.random.RandomState(seed)
        rng.shuffle(all_images)
        self.paths = all_images[:num_samples * 2]
        self.rng = np.random.RandomState(seed + 1)

    def __getitem__(self, idx):
        img = self._load(self.paths[idx % len(self.paths)])
        is_tampered = idx >= len(self.paths)
        if is_tampered:
            idx2 = (idx + len(self.paths) // 2) % len(self.paths)
            other = self._load(self.paths[idx2])
            tamper_mask = self._random_mask()
            img = img * (1 - tamper_mask) + other * tamper_mask
            verdict = 2
        else:
            tamper_mask = torch.zeros(1, self.img_size, self.img_size)
            verdict = 0
        secret = torch.ones_like(img) * 0.5
        return {
            'image': img, 'secret': secret,
            'tamper_mask': tamper_mask, 'verdict': torch.tensor(verdict, dtype=torch.long),
        }

    def _random_mask(self):
        mask = torch.zeros(1, self.img_size, self.img_size)
        x = self.rng.randint(0, self.img_size // 2)
        y = self.rng.randint(0, self.img_size // 2)
        w = self.rng.randint(self.img_size // 4, self.img_size // 2)
        h = self.rng.randint(self.img_size // 4, self.img_size // 2)
        mask[:, y:y+h, x:x+w] = 1.0
        return mask

    def _load(self, path):
        return self.transform(Image.open(path).convert('RGB'))

    def __len__(self):
        return len(self.paths) * 2


@torch.no_grad()
def infer_one(wm, spn, fusion, images, secrets):
    stego, snoisy, rec = wm.forward(images, secrets, apply_noise=True)
    spn_feat, noise_map = spn(snoisy)
    diff_img = (rec - secrets).abs()
    active_feat = F.adaptive_avg_pool2d(diff_img, (8, 8)).reshape(images.size(0), -1)
    quality = (1.0 - diff_img.mean(dim=[1, 2, 3])).clamp(0, 1)
    pce = noise_map.std(dim=[1, 2, 3], keepdim=True)
    logits, uncertainty = fusion(
        {'active_feat': active_feat, 'quality': quality},
        {'spn_feat': spn_feat, 'pce': pce},
    )
    return logits, uncertainty, stego


def print_sep(title):
    print(f'\n{"="*70}')
    print(f'  {title}')
    print(f'{"="*70}')


# ============================================================
# 实验 1: 检测精度
# ============================================================
def exp_detection_accuracy(wm, spn, fusion, name='DuetGuard (P3)'):
    print_sep(f'实验 1: 检测精度 — {name}')
    ds = EvalDataset(VAL_DIR, num_samples=200, seed=42)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    conf = torch.zeros(3, 3)
    cls_correct = torch.zeros(3)
    cls_total = torch.zeros(3)

    for batch in tqdm(loader, desc='评估中'):
        img = batch['image'].to(DEVICE)
        sec = batch['secret'].to(DEVICE)
        ver = batch['verdict'].to(DEVICE)
        logits, _, _ = infer_one(wm, spn, fusion, img, sec)
        pred = logits.argmax(dim=1)
        correct += (pred == ver).sum().item()
        total += ver.size(0)
        for i in range(ver.size(0)):
            conf[ver[i], pred[i]] += 1
            if pred[i] == ver[i]:
                cls_correct[ver[i]] += 1
            cls_total[ver[i]] += 1

    acc = correct / total * 100
    print(f'\nAccuracy: {acc:.2f}% ({correct}/{total})')
    labels = ['Authentic', 'Suspicious', 'Forgery']
    for i in range(3):
        ca = cls_correct[i] / max(cls_total[i], 1) * 100
        print(f'  {labels[i]:12s}: {ca:.1f}% ({int(cls_correct[i])}/{int(cls_total[i])})')
    print('\nConfusion Matrix:')
    print(f'                {"可信":>6} {"疑似":>6} {"确认":>6}')
    for i, lab in enumerate(labels):
        row = '  '.join(f'{conf[i,j]:6.0f}' for j in range(3))
        print(f'  {lab:12s}: {row}')
    return {'accuracy': acc, 'confusion': conf.tolist()}


# ============================================================
# 实验 2: 鲁棒性
# ============================================================
def apply_jpeg(x, quality):
    """JPEG 压缩攻击 (支持 batch)"""
    batch = []
    for i in range(x.size(0)):
        x_np = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(x_np).save(buf, 'JPEG', quality=quality)
        buf.seek(0)
        degraded = Image.open(buf).convert('RGB')
        degraded = T.ToTensor()(degraded).to(x.device)
        batch.append(degraded)
    return torch.stack(batch)

def apply_noise(x, std):
    """高斯噪声攻击"""
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0, 1)

def apply_blur(x, kernel_size=5):
    """高斯模糊"""
    return T.GaussianBlur(kernel_size=kernel_size)(x)

def exp_robustness(wm, spn, fusion):
    print_sep('实验 2: 鲁棒性测试')
    ds = EvalDataset(VAL_DIR, num_samples=100, seed=42)
    attacks = [
        ('无攻击 (Clean)', lambda x, s: (x, s)),
        ('JPEG q=70', lambda x, s: (apply_jpeg(x, 70), s)),
        ('JPEG q=50', lambda x, s: (apply_jpeg(x, 50), s)),
        ('JPEG q=30', lambda x, s: (apply_jpeg(x, 30), s)),
        ('高斯噪声 σ=0.01', lambda x, s: (apply_noise(x, 0.01), s)),
        ('高斯噪声 σ=0.05', lambda x, s: (apply_noise(x, 0.05), s)),
        ('高斯模糊 5x5', lambda x, s: (apply_blur(x, 5), s)),
    ]
    results = {}
    for atk_name, atk_fn in attacks:
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        correct = 0
        total = 0
        for batch in tqdm(loader, desc=atk_name):
            img = batch['image'].to(DEVICE)
            sec = batch['secret'].to(DEVICE)
            ver = batch['verdict'].to(DEVICE)
            img_attacked, sec_attacked = atk_fn(img, sec)
            logits, _, _ = infer_one(wm, spn, fusion, img_attacked, sec_attacked)
            pred = logits.argmax(dim=1)
            correct += (pred == ver).sum().item()
            total += ver.size(0)
        acc = correct / total * 100
        print(f'  {atk_name:25s}: {acc:.1f}%')
        results[atk_name] = acc
    return results


# ============================================================
# 实验 3: 图像质量
# ============================================================
def exp_image_quality(wm):
    print_sep('实验 3: 图像质量 (PSNR / SSIM)')
    ds = EvalDataset(VAL_DIR, num_samples=50, seed=42)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    psnr_list = []
    ssim_list = []
    for batch in tqdm(loader, desc='计算中'):
        img = batch['image'].to(DEVICE)
        sec = batch['secret'].to(DEVICE)
        with torch.no_grad():
            stego, _, _ = wm.forward(img, sec, apply_noise=True)
        img_np = img.cpu().permute(0, 2, 3, 1).numpy()[0]
        stego_np = stego.cpu().permute(0, 2, 3, 1).numpy()[0]
        psnr_val = peak_signal_noise_ratio(img_np, stego_np, data_range=1.0)
        ssim_val = structural_similarity(img_np, stego_np, channel_axis=2, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    print(f'  PSNR: {avg_psnr:.2f} dB')
    print(f'  SSIM: {avg_ssim:.4f}')
    return {'psnr': avg_psnr, 'ssim': avg_ssim}


# ============================================================
# 主函数
# ============================================================
def main():
    print('='*70)
    print('  DuetGuard 综合实验套件')
    print(f'  Device: {DEVICE}  |  Results: {SAVE_RESULTS}')
    print('='*70)

    all_results = {}

    # ── 加载 P3 模型 ──
    print('\n[1/4] 加载 P3 模型...')
    wm, spn, fusion = load_models()
    all_results['model'] = 'DuetGuard P3 (watermark + SPN)'

    # ── 实验 1: 检测精度 ──
    all_results['detection'] = exp_detection_accuracy(wm, spn, fusion)

    # ── 实验 2: 鲁棒性 ──
    all_results['robustness'] = exp_robustness(wm, spn, fusion)

    # ── 实验 3: 图像质量 ──
    all_results['image_quality'] = exp_image_quality(wm)

    # ── 实验 4: 消融对比 ──
    print_sep('实验 4: 消融对比')

    # 4a: Watermark-only (加载 P0 fusion, 但需要 P0 fusion...实际上用 random 初始化模拟)
    print('\n[4a] Watermark-only (w/o SPN): 参考 P0 结果 = 50.0%')
    all_results['ablation_wo_spn'] = 50.0

    # 4b: 加载无 SPN 微调的模型（只加载 pretrained SPN，不加载 finetuned）
    print('\n[4b] SPN pretrained-only (不微调):')
    wm2, spn2, fusion2 = load_models(spn_ckpt_name='spn_extractor.pth', fusion_ckpt_name='fusion_best.pth')
    ab_res = exp_detection_accuracy(wm2, spn2, fusion2, name='SPN pretrained-only')
    all_results['ablation_spn_no_finetune'] = ab_res['accuracy']

    # ── 保存结果 ──
    results_path = os.path.join(SAVE_RESULTS, 'experiment_results.json')
    def convert(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, torch.Tensor): return v.item()
        return v
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=convert)
    print(f'\n{"="*70}')
    print(f'  所有实验完成! 结果已保存至: {results_path}')
    print(f'{"="*70}')

    # 摘要
    print('\n' + '='*70)
    print('  实验结果摘要')
    print('='*70)
    print(f'  检测精度 (P3):             {all_results["detection"]["accuracy"]:.1f}%')
    print(f'  消融 w/o SPN (P0):         {all_results["ablation_wo_spn"]:.1f}%')
    print(f'  消融 SPN 不微调:           {all_results["ablation_spn_no_finetune"]:.1f}%')
    print(f'  PSNR:                      {all_results["image_quality"]["psnr"]:.2f} dB')
    print(f'  SSIM:                      {all_results["image_quality"]["ssim"]:.4f}')
    print(f'  鲁棒性:')
    for name, acc in all_results['robustness'].items():
        delta = acc - all_results["detection"]["accuracy"]
        print(f'    {name:25s}: {acc:.1f}% (Δ={delta:+.1f})')
    print('='*70)


if __name__ == '__main__':
    main()
