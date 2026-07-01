"""
Joint training: OmniGuard (frozen) + SPN (trainable in P3) + FusionLayer
Phases: P0=quick_test, P1=wm_warmup, P2=spn_pretrain, P3=joint_train
"""
import sys, os, time, gc
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from apjf_src.models.adapter import WatermarkBranch
from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.models.fusion_layer import FusionLayer
from apjf_src.data.dataset import JointTrainDataset
from apjf_src.losses import JointLoss
from apjf_src.joint_config import *


def build_model(device, phase='P0'):
    wm = WatermarkBranch(OMNIGUARD_CKPT, device)
    for p in wm.parameters():
        p.requires_grad = False
    wm.eval()

    spn = SPNExtractor(fp_dim=FP_DIM, base_ch=64, num_blocks=2, expand_ratio=4).to(device)
    if os.path.exists(SPN_CKPT):
        spn.load_state_dict(torch.load(SPN_CKPT, map_location='cpu', weights_only=True))
        print(f'[SPN] Loaded pretrained')

    if phase in ('P3',):
        for p in spn.parameters():
            p.requires_grad = True
        spn.train()
        print(f'[SPN] UNFROZEN (trainable)')
    else:
        for p in spn.parameters():
            p.requires_grad = False
        spn.eval()

    fusion = FusionLayer(fp_dim=FP_DIM, act_feat_dim=ACT_FEAT_DIM, num_classes=NUM_CLASSES).to(device)

    # 断点续训: 自动找最新的 checkpoint
    if RESUME and phase in ('P3',):
        ckpts = [f for f in os.listdir(SAVE_DIR) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
        if ckpts:
            latest = sorted(ckpts)[-1]
            ckpt_path = os.path.join(SAVE_DIR, latest)
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            fusion.load_state_dict(ckpt['fusion'])
            if ckpt['spn'] is not None:
                spn.load_state_dict(ckpt['spn'])
            print(f'[RESUME] Loaded {latest} (epoch {ckpt["epoch"]})')
        else:
            fusion_ckpt = os.path.join(SAVE_DIR, 'fusion_best.pth')
            spn_ckpt = os.path.join(SAVE_DIR, 'spn_finetuned_best.pth')
            if os.path.exists(fusion_ckpt) and os.path.exists(spn_ckpt):
                fusion.load_state_dict(torch.load(fusion_ckpt, map_location='cpu', weights_only=True))
                spn.load_state_dict(torch.load(spn_ckpt, map_location='cpu', weights_only=True))
                print(f'[RESUME] Loaded fusion_best + spn_finetuned_best')
            else:
                print(f'[RESUME] No checkpoint found, starting from scratch')

    return wm, spn, fusion


def build_optimizers(spn, fusion, phase):
    params = [{'params': fusion.parameters(), 'lr': LR_FUSION}]
    if phase in ('P3',):
        params.append({'params': spn.parameters(), 'lr': LR_SPN})
    return torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def extract_watermark_evidence(wm, images, secrets):
    stego, stego_noisy, recovered = wm.forward(images, secrets, apply_noise=True)
    diff_map_active = wm.extract_diff_map(stego_noisy, secrets)
    quality = 1.0 - (recovered - secrets).abs().mean(dim=[1, 2, 3])
    quality = quality.clamp(0, 1)
    diff_img = (recovered - secrets).abs()
    active_feat = torch.nn.functional.adaptive_avg_pool2d(diff_img, (8, 8))
    active_feat = active_feat.reshape(images.size(0), -1)
    return stego, stego_noisy, recovered, diff_map_active, quality, active_feat


def train_epoch(loader, wm, spn, fusion, opt, criterion, scaler, device, epoch, writer, step, phase):
    if phase in ('P3',):
        spn.train()
    fusion.train()
    total_loss = total_fusion = total_fp = total_consist = 0
    n_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        secrets = batch['secret'].to(device)
        tamper_mask = batch['tamper_mask'].to(device)
        verdict = batch['verdict'].to(device)

        opt.zero_grad()

        with torch.no_grad():
            stego, snoisy, rec, diff_active, quality, active_feat = \
                extract_watermark_evidence(wm, images, secrets)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # SPN on clean image (reference for L_fp)
            spn_ref_feat, noise_map_clean = spn(images)
            # SPN on watermarked image
            spn_feat, noise_map = spn(snoisy)

            pce = noise_map.std(dim=[1, 2, 3], keepdim=True)

            active_ev = {'active_feat': active_feat, 'quality': quality}
            passive_ev = {'spn_feat': spn_feat, 'pce': pce}
            logits, uncertainty = fusion(active_ev, passive_ev)

            evidence = {
                'stego': stego, 'stego_noisy': snoisy, 'recovered_secret': rec,
                'diff_map_active': diff_active, 'spn_feat': spn_feat, 'noise_map': noise_map,
                'fusion_logits': logits,
            }
            targets = {
                'cover': images, 'secret': secrets,
                'spn_ref': spn_ref_feat,  # clean image SPN feature as reference
                'tamper_mask': tamper_mask, 'verdict': verdict,
            }
            losses = criterion(evidence, targets)

        scaler.scale(losses['L_total']).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += losses['L_total'].item()
        total_fusion += losses['L_fusion'].item()
        total_fp += losses['L_fp'].item()
        total_consist += losses['L_consist'].item()
        n_batches += 1

        if step % 10 == 0:
            writer.add_scalar('train/L_total', losses['L_total'].item(), step)
            writer.add_scalar('train/L_fusion', losses['L_fusion'].item(), step)
            writer.add_scalar('train/L_fp', losses['L_fp'].item(), step)
            writer.add_scalar('train/L_consist', losses['L_consist'].item(), step)
        step += 1

    n = max(n_batches, 1)
    return total_loss / n, total_fusion / n, total_fp / n, total_consist / n


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Phase: {PHASE}')

    data_dir = COCO_MINI_DIR if PHASE == 'P0' else TRAIN_DIR
    n_samples = 500 if PHASE == 'P0' else 10000
    print(f'Data: {data_dir} ({n_samples} samples)')

    wm, spn, fusion = build_model(device, PHASE)
    t_params = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    t_params += sum(p.numel() for p in spn.parameters() if p.requires_grad)
    print(f'Trainable params: {t_params:,}')

    ds = JointTrainDataset(data_dir, img_size=IMAGE_SIZE, num_samples=n_samples)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True)
    print(f'Dataset: {len(ds)} images, {len(loader)} batches/epoch')

    criterion = JointLoss(lambda_qual=LAMBDA_QUAL, lambda_fp=LAMBDA_FP, lambda_consist=LAMBDA_CONSIST)
    opt = build_optimizers(spn, fusion, PHASE)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    writer = SummaryWriter(os.path.join(LOG_DIR, f'joint_{PHASE}'))

    epochs = 10 if PHASE == 'P0' else EPOCHS

    start_epoch = 0
    best_loss = float('inf')
    if RESUME and PHASE in ('P3',):
        ckpts = [f for f in os.listdir(SAVE_DIR) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
        if ckpts:
            latest = sorted(ckpts)[-1]
            ckpt = torch.load(os.path.join(SAVE_DIR, latest), map_location='cpu', weights_only=True)
            start_epoch = ckpt['epoch']
            opt.load_state_dict(ckpt['optimizer'])
            best_loss = ckpt['best_loss']
            print(f'[RESUME] Continuing from checkpoint epoch {start_epoch} (best_loss={best_loss:.4f})')
        else:
            start_epoch = RESUME_EPOCH

    if start_epoch > 0:
        print(f'Resuming from epoch {start_epoch+1}/{epochs} ({epochs - start_epoch} remaining)...')
    else:
        print(f'Training {epochs} epochs...')

    t0 = time.time()

    for epoch in range(start_epoch, epochs):
        loss, fusion_loss, fp_loss, consist_loss = train_epoch(
            loader, wm, spn, fusion, opt, criterion, scaler,
            device, epoch, writer, epoch * len(loader), PHASE,
        )
        elapsed = (time.time() - t0) / 60
        current = epoch + 1
        print(f'E{current:3d}/{epochs} | Loss:{loss:.4f} CE:{fusion_loss:.4f} '
              f'FP:{fp_loss:.4f} Cons:{consist_loss:.4f} | {elapsed:.1f}m')

        writer.add_scalar('epoch/loss', loss, epoch)
        writer.add_scalar('epoch/fusion_ce', fusion_loss, epoch)
        writer.add_scalar('epoch/spn_fp', fp_loss, epoch)
        writer.add_scalar('epoch/consist', consist_loss, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(fusion.state_dict(), os.path.join(SAVE_DIR, 'fusion_best.pth'))
            if PHASE in ('P3',):
                torch.save(spn.state_dict(), os.path.join(SAVE_DIR, 'spn_finetuned_best.pth'))

        if (current) % 5 == 0:
            torch.save({
                'epoch': current,
                'fusion': fusion.state_dict(),
                'spn': spn.state_dict() if PHASE in ('P3',) else None,
                'optimizer': opt.state_dict(),
                'best_loss': best_loss,
            }, os.path.join(SAVE_DIR, f'checkpoint_epoch{current:03d}.pth'))

    torch.save(fusion.state_dict(), os.path.join(SAVE_DIR, 'fusion_final.pth'))
    if PHASE in ('P3',):
        torch.save(spn.state_dict(), os.path.join(SAVE_DIR, 'spn_finetuned.pth'))
    writer.close()
    print(f'Done in {(time.time()-t0)/60:.1f}m. Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
