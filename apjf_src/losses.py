"""
五项联合损失函数
- L_msg: 消息损失 (BCE for bits + MSE for loc watermark)
- L_qual: 图像质量损失 (MSE + SSIM)
- L_fp: SPN 指纹一致性损失 (新增)
- L_consist: 双模式定位掩膜一致性损失 (核心创新)
- L_fusion: 融合分类交叉熵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointLoss(nn.Module):
    """
    主动-被动联合训练损失
    """

    def __init__(self, lambda_qual=0.1, lambda_fp=0.5, lambda_consist=0.3):
        super().__init__()
        self.lambda_qual = lambda_qual
        self.lambda_fp = lambda_fp
        self.lambda_consist = lambda_consist

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: {
                'stego': 含水印图,
                'stego_noisy': 加噪后的含水印图,
                'recovered_secret': 恢复的秘密图像,
                'diff_map_active': 主动定位掩膜,
                'spn_feat': 提取的SPN指纹,
                'noise_map': 噪声残差图,
                'fusion_logits': 融合分类 logits,
            }
            targets: {
                'cover': 原始图,
                'secret': 原始秘密图像（定位水印）,
                'spn_ref': 参考SPN指纹,
                'tamper_mask': 真值篡改掩膜,
                'verdict': 真值分类标签,
            }
        Returns:
            dict of losses
        """
        # L1: 消息/定位损失
        L_secret = self.mse(outputs['recovered_secret'], targets['secret'])

        # L2: 图像质量损失
        L_qual = self.mse(outputs['stego'], targets['cover'])
        # 简化版：只用 MSE，SSIM 太慢

        # L3: SPN 指纹一致性损失（可选，SPN frozen 时可设为 0）
        if 'spn_ref' in targets and targets['spn_ref'] is not None:
            L_fp = self.mse(outputs['spn_feat'], targets['spn_ref'])
        else:
            L_fp = torch.tensor(0.0, device=outputs['stego'].device)

        # L4: 主动-被动掩膜一致性损失（核心创新）
        if 'tamper_mask' in targets and targets['tamper_mask'] is not None:
            # 有真值掩膜时，使用真值
            L_consist_active = self.mse(
                outputs['diff_map_active'], targets['tamper_mask']
            )
            # 被动掩膜也应该接近真值
            passive_mask = outputs['noise_map'].mean(dim=1, keepdim=True)
            passive_mask = torch.sigmoid(passive_mask - passive_mask.mean())
            L_consist = self.mse(passive_mask, targets['tamper_mask'])
        else:
            # 无真值时，让主动和被动掩膜相互一致
            passive_mask = outputs['noise_map'].mean(dim=1, keepdim=True)
            passive_mask = torch.sigmoid(passive_mask - passive_mask.mean())
            L_consist = self.mse(
                outputs['diff_map_active'], passive_mask
            )

        # L5: 融合分类损失
        if 'verdict' in targets and targets['verdict'] is not None:
            L_fusion = F.cross_entropy(
                outputs['fusion_logits'], targets['verdict']
            )
        else:
            L_fusion = torch.tensor(0.0, device=outputs['stego'].device)

        # 总损失
        L_total = (
            L_secret
            + self.lambda_qual * L_qual
            + self.lambda_fp * L_fp
            + self.lambda_consist * L_consist
            + L_fusion
        )

        return {
            'L_secret': L_secret,
            'L_qual': L_qual,
            'L_fp': L_fp,
            'L_consist': L_consist,
            'L_fusion': L_fusion,
            'L_total': L_total,
        }
