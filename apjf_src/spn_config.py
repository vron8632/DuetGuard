"""
SPN 指纹提取器配置
"""

# 数据集
DATA_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/coco/train2017'
COCO_MINI_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/data/coco_mini'

# 模型
FP_DIM = 128
BASE_CH = 64
NUM_BLOCKS = 4

# 训练
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5
MARGIN = 1.0
TEMPERATURE = 0.07
USE_INFONCE = True

# 图像
IMG_SIZE = 256
NOISE_STD = 0.01

# 保存
SAVE_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/weights'
LOG_DIR = '/media/oyp/数据/Projects/042_image_forensic/DuetGuard/logs'
