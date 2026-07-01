"""
主动-被动融合决策层
输入：水印证据 (BER) + SPN 指纹 (128d) + 置信度
输出：3 级认证结论 [可信, 疑似篡改, 确认篡改] + 不确定性分数

设计原则：
  - 当主动水印和被动 SPN 都通过时 -> 可信
  - 当只有一方异常时 -> 疑似篡改
  - 当两方都异常时 -> 确认篡改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """
    主动-被动融合决策层
    OmniGuard 恢复的是图像（非比特），使用 pooled diff map 作为主动证据
    输入证据维度：act_feat_dim + fp_dim + 2
    """

    def __init__(self, fp_dim=128, act_feat_dim=192, num_classes=3):
        super().__init__()
        evidence_dim = act_feat_dim + fp_dim + 2  # +2 for PCE + quality

        self.fusion = nn.Sequential(
            nn.Linear(evidence_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.uncertainty = nn.Sequential(
            nn.Linear(evidence_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, active_evidence, passive_evidence):
        """
        Args:
            active_evidence: {
                'active_feat': (B, act_feat_dim) pooled diff map features,
                'quality': (B, 1) 主动水印恢复质量
            }
            passive_evidence: {
                'spn_feat': (B, fp_dim) SPN 指纹,
                'pce': (B, 1) 峰值相关能量
            }
        Returns:
            logits: (B, num_classes) 分类 logits
            uncertainty: (B, 1) 不确定性分数 [0, 1]
        """
        evidence = torch.cat([
            active_evidence['active_feat'],
            passive_evidence['spn_feat'],
            passive_evidence['pce'].view(-1, 1),
            active_evidence['quality'].view(-1, 1),
        ], dim=1)

        logits = self.fusion(evidence)
        uncertainty = torch.sigmoid(self.uncertainty(evidence))

        return logits, uncertainty
