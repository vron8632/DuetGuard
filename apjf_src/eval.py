"""
DuetGuard evaluation: test fusion model on synthetic tampered images
Usage: conda run -n apjf python apjf_src/eval.py
"""
import sys, os, torch
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
from torch.utils.data import DataLoader
from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.data.dataset import JointTrainDataset
from apjf_src.joint_config import *

device = 'cuda'

wm = WatermarkBranch(OMNIGUARD_CKPT, device).eval()
for p in wm.parameters(): p.requires_grad = False

spn = SPNExtractor(fp_dim=FP_DIM, base_ch=64, num_blocks=2, expand_ratio=4).to(device)
# 优先加载 P3 微调后的 SPN，否则用预训练权重
spn_ckpt = os.path.join(SAVE_DIR, 'spn_finetuned_best.pth')
if os.path.exists(spn_ckpt):
    spn.load_state_dict(torch.load(spn_ckpt, map_location='cpu', weights_only=True))
    print(f'Loaded SPN from {spn_ckpt} (P3 finetuned)')
else:
    spn.load_state_dict(torch.load(SPN_CKPT, map_location='cpu', weights_only=True))
    print(f'Loaded SPN from {SPN_CKPT} (pretrained only)')
spn.eval()

fusion = FusionLayer(fp_dim=FP_DIM, act_feat_dim=ACT_FEAT_DIM, num_classes=NUM_CLASSES).to(device)
fusion_ckpt = os.path.join(SAVE_DIR, 'fusion_best.pth')
if os.path.exists(fusion_ckpt):
    fusion.load_state_dict(torch.load(fusion_ckpt, map_location='cpu', weights_only=True))
    print(f'Loaded fusion from {fusion_ckpt}')
else:
    print('WARNING: no fusion checkpoint found')
fusion.eval()

ds = JointTrainDataset(VAL_DIR, img_size=IMAGE_SIZE, num_samples=200)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, drop_last=False)

correct = 0
total = 0
conf_matrix = torch.zeros(3, 3)
class_correct = torch.zeros(3)
class_total = torch.zeros(3)
all_uncertainties = []

with torch.no_grad():
    for batch in loader:
        images = batch['image'].to(device)
        secrets = batch['secret'].to(device)
        verdict = batch['verdict'].to(device)

        stego, snoisy, rec = wm.forward(images, secrets, apply_noise=True)
        spn_feat, noise_map = spn(snoisy)
        diff_img = (rec - secrets).abs()
        active_feat = torch.nn.functional.adaptive_avg_pool2d(diff_img, (8, 8)).reshape(images.size(0), -1)
        quality = 1.0 - diff_img.mean(dim=[1, 2, 3])
        quality = quality.clamp(0, 1)
        pce = noise_map.std(dim=[1, 2, 3], keepdim=True)

        active_ev = {'active_feat': active_feat, 'quality': quality}
        passive_ev = {'spn_feat': spn_feat, 'pce': pce}
        logits, uncertainty = fusion(active_ev, passive_ev)
        pred = logits.argmax(dim=1)

        correct += (pred == verdict).sum().item()
        total += verdict.size(0)
        all_uncertainties.append(uncertainty.cpu())

        for i in range(verdict.size(0)):
            conf_matrix[verdict[i], pred[i]] += 1
            if pred[i] == verdict[i]:
                class_correct[verdict[i]] += 1
            class_total[verdict[i]] += 1

uncertainties = torch.cat(all_uncertainties)
acc = correct / max(total, 1) * 100
print(f'\n=== DuetGuard Evaluation ===')
print(f'Samples: {total}')
print(f'Accuracy: {acc:.1f}% ({correct}/{total})')
print(f'\nPer-class Accuracy:')
labels = ['Authentic(0)', 'Suspicious(1)', 'Forgery(2)']
for i, label in enumerate(labels):
    ca = class_correct[i] / max(class_total[i], 1) * 100
    print(f'  {label:15s}: {ca:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')
print(f'\nConfusion Matrix (rows=true, cols=pred):')
print(f'               可信    疑似    确认')
for i, label in enumerate(labels):
    row = '  '.join(f'{conf_matrix[i,j]:6.0f}' for j in range(3))
    print(f'  {label:15s}: {row}')
print(f'\nUncertainty: mean={uncertainties.mean().item():.4f} '
      f'std={uncertainties.std().item():.4f} '
      f'min={uncertainties.min().item():.4f} '
      f'max={uncertainties.max().item():.4f}')
print(f'  Class0(可信):   {uncertainties[:class_total[0].int()].mean().item():.4f}')
print(f'  Class1(疑似):   {uncertainties[class_total[0].int():class_total[0].int()+class_total[1].int()].mean().item():.4f}')
print(f'  Class2(确认篡改): {uncertainties[class_total[0].int()+class_total[1].int():].mean().item():.4f}')
