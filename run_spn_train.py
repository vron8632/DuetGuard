"""Standalone SPN training - clean run"""
import torch, sys, os, time, gc
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
from torch.utils.data import DataLoader
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.data.spn_dataset import SPNTrainDataset, SPNContrastiveLoss

device = torch.device('cuda')
print(f'Initial GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB allocated')

# Data
ds = SPNTrainDataset('/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/coco_mini',
                     num_samples=500, noise_std=0.01, img_size=256)
loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# Model
model = SPNExtractor(fp_dim=128, base_ch=64, num_blocks=2, expand_ratio=2).to(device)
print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
print(f'After model: {torch.cuda.memory_allocated()/1e6:.0f}MB')

criterion = SPNContrastiveLoss()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

best_loss = float('inf')
t0 = time.time()
for epoch in range(20):
    tl = ts = tn = 0
    for batch in loader:
        a = batch['anchor'].to(device)
        p = batch['positive'].to(device)
        n = batch['negative'].to(device)
        fpa, _ = model(a)
        fpp, _ = model(p)
        fpn, _ = model(n)
        loss, m = criterion(fpa, fpp, fpn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tl += loss.item()
        ts += m['sim_pos'].item()
        tn += m['sim_neg'].item()
    nb = len(loader)
    avg_l = tl / nb
    print(f'E{epoch+1:2d}: Loss={avg_l:.4f} S+={ts/nb:.3f} S-={tn/nb:.3f} D={(ts-tn)/nb:.3f} Mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB')
    if avg_l < best_loss:
        best_loss = avg_l
        os.makedirs('/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights', exist_ok=True)
        torch.save(model.state_dict(), '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights/spn_extractor_best.pth')

# Validation
model.eval()
sp_av = sn_av = 0.0
with torch.no_grad():
    for i in range(50):
        a = ds[i]['anchor'].unsqueeze(0).to(device)
        p = ds[i]['positive'].unsqueeze(0).to(device)
        n = ds[(i+20) % len(ds)]['negative'].unsqueeze(0).to(device)
        fpa, _ = model(a)
        fpp, _ = model(p)
        fpn, _ = model(n)
        sp_av += torch.nn.functional.cosine_similarity(fpa, fpp).item()
        sn_av += torch.nn.functional.cosine_similarity(fpa, fpn).item()
sp_av /= 50
sn_av /= 50
print(f'\nValidation (50 pairs): Same={sp_av:.4f} Diff={sn_av:.4f} Gap={sp_av-sn_av:.4f}')
print(f'SPN working: {"YES" if sp_av > sn_av else "NO"}')

torch.save(model.state_dict(), '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights/spn_extractor.pth')
print(f'Done in {(time.time()-t0)/60:.1f}m | Best loss: {best_loss:.4f}')
