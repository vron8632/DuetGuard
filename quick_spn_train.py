"""Quick SPN training test"""
import sys, os, time
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
from torch.utils.data import DataLoader
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.data.spn_dataset import SPNTrainDataset, SPNContrastiveLoss

device = torch.device('cuda')
ds = SPNTrainDataset('/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/coco_mini',
                     num_samples=200, noise_std=0.01)
loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True)

model = SPNExtractor(fp_dim=128).to(device)
criterion = SPNContrastiveLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f'Training {len(ds)} samples x 10 epochs (batch=8)...')
model.train()
t0 = time.time()
for epoch in range(10):
    total_loss = 0
    for batch in loader:
        a = batch['anchor'].to(device)
        p = batch['positive'].to(device)
        n = batch['negative'].to(device)
        fpa, _ = model(a); fpp, _ = model(p); fpn, _ = model(n)
        loss, m = criterion(fpa, fpp, fpn)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    avg = total_loss / len(loader)
    print(f'Epoch {epoch+1}: Loss={avg:.4f} Sim+={m["sim_pos"].item():.3f} Sim-={m["sim_neg"].item():.3f}')

model.eval()
with torch.no_grad():
    a = ds[0]['anchor'].unsqueeze(0).to(device)
    p = ds[0]['positive'].unsqueeze(0).to(device)
    n = ds[10]['negative'].unsqueeze(0).to(device)
    fpa, _ = model(a); fpp, _ = model(p); fpn, _ = model(n)
    sp = torch.nn.functional.cosine_similarity(fpa, fpp).item()
    sn = torch.nn.functional.cosine_similarity(fpa, fpn).item()
    print(f'\nValidation: Same={sp:.3f} Diff={sn:.3f} Diff={sp-sn:.3f}')
    print(f'Result: {"PASSED" if sp > sn else "FAILED"}')

os.makedirs('/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights', exist_ok=True)
torch.save(model.state_dict(), '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights/spn_quick_test.pth')
print(f'Done in {time.time()-t0:.1f}s')
