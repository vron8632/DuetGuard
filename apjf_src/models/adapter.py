"""
OmniGuard 模型适配器
包装 OmniGuard 的 Model 类，提供:
  - encode(cover, secret) -> stego_image
  - decode(stego_image) -> recovered_secret
  - forward(cover, secret, apply_noise) -> (stego, noisy_stego, recovered_secret)

不修改 OmniGuard 源码，仅通过 sys.path 导入
"""

import sys
import os

OMNIGUARD_PATH = os.path.join(os.path.dirname(__file__), '../../baselines/OmniGuard')
sys.path.insert(0, OMNIGUARD_PATH)

import torch
import torch.nn as nn

import modules.Unet_common as common
from model_invert import Model as OmniGuardModel
import config as oc


class WatermarkBranch(nn.Module):
    """
    OmniGuard 主动水印分支包装器
    基于 Hinet 16 可逆块 + BitModule (TrustMark) 架构
    """

    def __init__(self, checkpoint_path=None, device='cuda'):
        super().__init__()

        # OmniGuard Model.__init__ 会从相对路径加载 checkpoint，需要切换工作目录
        old_cwd = os.getcwd()
        os.chdir(OMNIGUARD_PATH)
        self.model = OmniGuardModel()
        os.chdir(old_cwd)

        # DWT/IWT 变换
        self.dwt = common.DWT()
        self.iwt = common.IWT()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)

        self.to(device)

    def _load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        new_state = {}
        for k, v in ckpt['net'].items():
            if 'tmp_var' in k:
                continue
            new_key = k[len('module.'):] if k.startswith('module.') else k
            new_state[new_key] = v
        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        if missing:
            print(f'[WatermarkBranch] Warning: {len(missing)} missing keys')
        if unexpected:
            print(f'[WatermarkBranch] Warning: {len(unexpected)} unexpected keys')

    def encode(self, cover, secret):
        """
        嵌入水印
        Args:
            cover: (B, 3, H, W) 原始图像，值域 [0, 1]
            secret: (B, 3, H, W) 秘密图像（定位水印），值域 [0, 1]
        Returns:
            stego: (B, 3, H, W) 含水印图像
        """
        with torch.no_grad():
            cover_f = self.dwt(cover)
            secret_f = self.dwt(secret)
            stego, _, _, _ = self.model(cover_f, secret_f)
        return stego

    def decode(self, stego):
        """
        提取水印
        Args:
            stego: (B, 3, H, W) 含水印图像，值域 [0, 1]
        Returns:
            recovered_secret: (B, 3, H, W) 恢复的秘密图像
        """
        with torch.no_grad():
            stego_f = self.dwt(stego)
            rev = self.model(stego_f, rev=True)
            secret_rev = rev.narrow(1, 0, 12)
            secret_rev = self.iwt(secret_rev)
        return secret_rev

    def forward(self, cover, secret, apply_noise=False):
        """
        完整的 hide + noise + reveal 流程
        Args:
            cover: (B, 3, H, W) 原始图像
            secret: (B, 3, H, W) 秘密图像
            apply_noise: 是否添加噪声层（用于训练）
        Returns:
            stego: watermark 嵌入后的图像
            stego_noisy: (可选) 噪声后的含水印图
            recovered_secret: 提取的秘密图像
        """
        with torch.no_grad():
            stego = self.encode(cover, secret)

        if apply_noise:
            stego_noisy = stego + torch.randn_like(stego) * 0.01
            stego_noisy = torch.clamp(stego_noisy, 0, 1)
        else:
            stego_noisy = stego

        with torch.no_grad():
            recovered_secret = self.decode(stego_noisy)

        return stego, stego_noisy, recovered_secret

    def extract_diff_map(self, stego, secret):
        """
        提取主动分支的差异图（用于定位篡改区域）
        Args:
            stego: (B, 3, H, W) 含水印图像
            secret: (B, 3, H, W) 原始秘密图像（定位水印）
        Returns:
            diff_map: (B, 1, H, W) 差异图，0=一致，1=篡改
        """
        recovered = self.decode(stego)
        diff = torch.abs(recovered - secret).mean(dim=1, keepdim=True)
        return diff
