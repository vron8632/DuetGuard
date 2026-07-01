"""
SPN 指纹提取器单独预训练
用 COCO 数据 + 自监督对比学习
"""

import sys
sys.path.insert(0, '/media/oyp/数据/Projects/042_image_forensic/DuetGuard')

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time

from apjf_src.models.spn_extractor import SPNExtractor
from apjf_src.data.spn_dataset import SPNTrainDataset, SPNContrastiveLoss
import apjf_src.spn_config as cfg


def main():
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'FP16: {device.type == "cuda"}')

    # 数据
    data_dir = cfg.COCO_MINI_DIR if os.path.isdir(cfg.COCO_MINI_DIR) else cfg.DATA_DIR
    dataset = SPNTrainDataset(
        data_dir,
        img_size=cfg.IMG_SIZE,
        noise_std=cfg.NOISE_STD,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 模型
    model = SPNExtractor(
        fp_dim=cfg.FP_DIM,
        base_ch=cfg.BASE_CH,
        num_blocks=cfg.NUM_BLOCKS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'SPNExtractor: {total_params:,} params ({trainable_params:,} trainable)')

    # 损失 + 优化器
    criterion = SPNContrastiveLoss(
        margin=cfg.MARGIN,
        temperature=cfg.TEMPERATURE,
        use_infonce=cfg.USE_INFONCE,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # TensorBoard
    writer = SummaryWriter(os.path.join(cfg.LOG_DIR, 'spn_train'))

    # 训练
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_loss = float('inf')
    step = 0
    start_time = time.time()

    for epoch in range(cfg.EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_sim_pos = 0
        epoch_sim_neg = 0

        for batch in loader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                fp_a, _ = model(anchor)
                fp_p, _ = model(positive)
                fp_n, _ = model(negative)

                loss, metrics = criterion(fp_a, fp_p, fp_n)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_sim_pos += metrics['sim_pos'].item()
            epoch_sim_neg += metrics['sim_neg'].item()

            if step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), step)
                writer.add_scalar('Sim/positive', metrics['sim_pos'].item(), step)
                writer.add_scalar('Sim/negative', metrics['sim_neg'].item(), step)
                writer.add_scalar('Sim/diff', (metrics['sim_pos'] - metrics['sim_neg']).item(), step)
            step += 1

        scheduler.step()

        n = len(loader)
        avg_loss = epoch_loss / n
        avg_sim_pos = epoch_sim_pos / n
        avg_sim_neg = epoch_sim_neg / n

        elapsed = time.time() - start_time
        print(f'Epoch {epoch+1:3d}/{cfg.EPOCHS} | '
              f'Loss: {avg_loss:.4f} | '
              f'Sim+: {avg_sim_pos:.4f} | '
              f'Sim-: {avg_sim_neg:.4f} | '
              f'Diff: {avg_sim_pos - avg_sim_neg:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.2e} | '
              f'Time: {elapsed/60:.1f}m')

        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/sim_pos', avg_sim_pos, epoch)
        writer.add_scalar('Epoch/sim_neg', avg_sim_neg, epoch)
        writer.add_scalar('Epoch/sim_diff', avg_sim_pos - avg_sim_neg, epoch)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'spn_extractor_best.pth'))
            print(f'  ✅ Best model saved (loss={best_loss:.4f})')

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'spn_extractor.pth'))
    writer.close()

    print(f'\n✅ Training complete! Total time: {(time.time()-start_time)/60:.1f}m')
    print(f'Best loss: {best_loss:.4f}')


if __name__ == '__main__':
    main()
