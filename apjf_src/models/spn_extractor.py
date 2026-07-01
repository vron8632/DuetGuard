"""
轻量化 SPN (Sensor Pattern Noise) 指纹提取器
基于 MobileNetV3 风格深度可分离卷积设计
输入: 图像 (B,3,H,W)
输出: 128d 指纹向量 + 像素级 SPN 残差图 (B,64,H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighPassFilter(nn.Module):
    """可学习的高通滤波器，替代传统小波去噪中的固定滤波器"""

    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        kernel = torch.tensor([[[[-1, -1, -1],
                                  [-1,  8, -1],
                                  [-1, -1, -1]]]]) / 8.0
        self.weight = nn.Parameter(kernel.repeat(out_ch, in_ch, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)


class DepthwiseResBlock(nn.Module):
    """深度可分离残差块 (内存优化版)"""

    def __init__(self, ch=64, expand_ratio=2):
        super().__init__()
        hidden = ch * expand_ratio

        self.conv1 = nn.Conv2d(ch, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dw(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn3(out)
        return identity + out


class SPNExtractor(nn.Module):
    """
    轻量化 SPN 指纹提取器
    输入: (B, 3, H, W) 图像
    输出:
      - fingerprint: (B, fp_dim) 紧凑指纹向量
      - noise_map: (B, 64, H, W) 像素级噪声残差图
    """

    def __init__(self, fp_dim=128, base_ch=48, num_blocks=2, expand_ratio=2):
        super().__init__()
        self.fp_dim = fp_dim

        # Stage 1: 高通滤波 -> 提取噪声残差
        self.high_pass = HighPassFilter(3, base_ch)

        # Stage 2: 深度可分离残差块 (轻量特征提取)
        blocks = []
        for _ in range(num_blocks):
            blocks.append(DepthwiseResBlock(base_ch, expand_ratio=4))
        self.res_blocks = nn.Sequential(*blocks)

        # Stage 3: 全局池化 -> 紧凑指纹
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fingerprint_head = nn.Sequential(
            nn.Linear(base_ch, fp_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(fp_dim * 2, fp_dim),
        )

        # Stage 4: 噪声残差图头 (用于像素级定位)
        self.noise_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
        )

    def forward(self, x):
        noise = self.high_pass(x)
        feat = self.res_blocks(noise)

        # 紧凑指纹
        feat_pooled = self.gap(feat).flatten(1)
        fingerprint = self.fingerprint_head(feat_pooled)

        # 噪声残差图
        noise_map = self.noise_head(feat)

        return fingerprint, noise_map

    def extract_fingerprint_only(self, x):
        """仅提取指纹向量 (用于快速相似度计算)"""
        fp, _ = self.forward(x)
        return fp
