"""
联合训练配置
"""

import os

# 路径
BASE_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard'
OMNIGUARD_CKPT = os.path.join(BASE_DIR, 'baselines/OmniGuard/checkpoint/model_checkpoint_01500.pt')
SPN_CKPT = os.path.join(BASE_DIR, 'weights/spn_extractor.pth')

TRAIN_DIR = os.path.join(BASE_DIR, 'data/coco/train2017')
VAL_DIR = os.path.join(BASE_DIR, 'data/coco/val2017')
COCO_MINI_DIR = os.path.join(BASE_DIR, 'data/coco_mini')

SAVE_DIR = os.path.join(BASE_DIR, 'weights')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 模型
FP_DIM = 128
ACT_FEAT_DIM = 192  # pooled diff map: 3 channels x 8x8 = 192
NUM_CLASSES = 3

# 训练超参数（16GB 适配）
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50            # P3: 50 epochs (大数据)
LR_WATERMARK = 5e-5   # 水印分支低学习率（微调）
LR_SPN = 1e-4          # SPN 提取器
LR_FUSION = 1e-3       # 融合层新模块用高学习率
WEIGHT_DECAY = 1e-5
FP16 = True

# 损失权重 (P3: 解冻 SPN, lambda_fp 启用)
LAMBDA_QUAL = 0.1       # 图像质量损失
LAMBDA_FP = 0.5         # SPN 指纹一致性损失
LAMBDA_CONSIST = 0.3    # 主动-被动一致性损失

# 训练阶段 (切换时改这里)
PHASE = 'P3'  # P0=快速验证, P1=水印预热, P2=SPN预训练, P3=联合训练

# 恢复训练 (P3 断点续训)
RESUME = True           # True=加载最近 checkpoint 续训
RESUME_EPOCH = 34       # 无 checkpoint 文件时用此值作为起始 epoch
