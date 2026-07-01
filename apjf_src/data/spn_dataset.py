"""
SPN 对比学习数据集
- 正样本对：同一图像 + 固定参考 SPN 模式 (模拟同相机)
- 负样本对：不同图像 (模拟不同相机)
- 三元组损失训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import os
from PIL import Image
from torchvision import transforms as T


def load_image(path, size=256):
    """加载并预处理图像"""
    img = Image.open(path).convert('RGB')
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
    ])
    return transform(img)


class SPNTrainDataset(Dataset):
    """
    SPN 对比学习训练数据集
    策略：
      正样本 = (I, I + fixed_noise) — 模拟同相机的 SPN 一致性
      负样本 = (I, J) — 不同图像 = 不同相机
    """

    def __init__(self, image_dir, num_samples=5000, noise_std=0.01, img_size=256):
        self.img_size = img_size
        self.noise_std = noise_std

        all_images = []
        for f in os.listdir(image_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(image_dir, f))
        random.seed(42)
        random.shuffle(all_images)
        self.paths = all_images[:min(num_samples, len(all_images))]
        print(f"SPN Dataset: {len(self.paths)} images from {image_dir}")

        # 固定参考 SPN 模式 (模拟相机指纹)
        self.ref_spn = torch.randn(3, img_size, img_size) * noise_std

    def __getitem__(self, idx):
        I = load_image(self.paths[idx], self.img_size)

        # 随机选另一张图作为负样本
        neg_idx = random.randint(0, len(self.paths) - 1)
        while neg_idx == idx:
            neg_idx = random.randint(0, len(self.paths) - 1)
        J = load_image(self.paths[neg_idx], self.img_size)

        # 正样本：原图 + 参考 SPN (模拟同相机)
        I_positive = I + self.ref_spn
        I_positive = torch.clamp(I_positive, 0, 1)

        return {
            'anchor': I,
            'positive': I_positive,
            'negative': J,
        }

    def __len__(self):
        return len(self.paths)


class SPNContrastiveLoss(nn.Module):
    """三元组损失 + InfoNCE 混合"""

    def __init__(self, margin=1.0, temperature=0.07, use_infonce=True):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.use_infonce = use_infonce

    def forward(self, fp_anchor, fp_positive, fp_negative):
        # L2 归一化
        fp_anchor = F.normalize(fp_anchor, dim=1)
        fp_positive = F.normalize(fp_positive, dim=1)
        fp_negative = F.normalize(fp_negative, dim=1)

        if self.use_infonce:
            # InfoNCE 损失 (更稳定)
            pos_sim = (fp_anchor * fp_positive).sum(dim=1) / self.temperature
            neg_sim = (fp_anchor * fp_negative).sum(dim=1) / self.temperature
            logits = torch.stack([pos_sim, neg_sim], dim=1)
            labels = torch.zeros(fp_anchor.size(0), dtype=torch.long, device=fp_anchor.device)
            loss = F.cross_entropy(logits, labels)
        else:
            # 三元组损失
            d_pos = F.pairwise_distance(fp_anchor, fp_positive)
            d_neg = F.pairwise_distance(fp_anchor, fp_negative)
            loss = torch.mean(F.relu(d_pos - d_neg + self.margin))

        # 计算监控指标
        with torch.no_grad():
            sim_pos = F.cosine_similarity(fp_anchor, fp_positive).mean()
            sim_neg = F.cosine_similarity(fp_anchor, fp_negative).mean()

        return loss, {'sim_pos': sim_pos, 'sim_neg': sim_neg}
