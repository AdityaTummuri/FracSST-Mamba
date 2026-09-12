"""Dual-Attention Segmentation (DA-Seg) network for Radar HAR.

Combines Physics-Aware spectral inputs with Mamba SSM sequence modeling and
dual attention features for temporal segmentation and classification of human activity signatures.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_block import MambaBlock


class ChannelAttention(nn.Module):
    """Channel/Frequency Attention module for radar feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        y = torch.mean(x, dim=1)  # (B, C)
        weights = self.fc(y).unsqueeze(1)  # (B, 1, C)
        return x * weights


class TemporalAttention(nn.Module):
    """Temporal Self-Attention module over spectrogram frame sequences."""

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class DASegModel(nn.Module):
    """Dual-Attention Segmentation Model for Radar HAR sequence labeling.

    Combines a 2D Conv spectral stem, Dual-Attention (Temporal + Channel), stacked
    Mamba SSM sequence blocks, and multi-task heads for sequence classification & temporal segmentation.

    Args:
        in_channels: Input channels (e.g. 1 for magnitude spectrogram).
        d_model: Internal feature dimension size.
        n_layers: Number of stacked Mamba sequence layers.
        num_classes: Number of target human activity classes.
        dropout: Dropout rate across intermediate layers.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_classes = num_classes

        # 1. 2D Convolutional Spectral Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, d_model // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model // 2),
            nn.SiLU(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )

        # Adaptive pool to collapse frequency dimension down to 1
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))


        # 2. Dual-Attention Modules
        self.channel_attn = ChannelAttention(d_model)
        self.temporal_attn = TemporalAttention(d_model)

        # 3. Stacked Mamba Selective State-Space Blocks
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 4. Multi-Task Heads
        # Frame-level temporal segmentation head: (B, T, num_classes)
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Global activity classification head: (B, num_classes)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self, x: torch.Tensor, return_segmentation: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for DA-Seg model.

        Args:
            x: Input tensor of shape (batch_size, in_channels, freq_bins, time_steps)
               or (batch_size, in_channels, time_steps, freq_bins) or (batch_size, freq_bins, time_steps).
            return_segmentation: If True, returns tuple (cls_logits, seg_logits). If False, returns cls_logits.

        Returns:
            cls_logits of shape (batch_size, num_classes)
            or (cls_logits, seg_logits) where seg_logits has shape (batch_size, time_steps, num_classes).
        """
        # Ensure 4D tensor: (B, C, F, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)

        # If shape is (B, C, T, F) where T < F, transpose to (B, C, F, T)
        b, c, dim1, dim2 = x.shape
        if dim1 < dim2 and dim1 < 32 and dim2 >= 32:
            x = x.transpose(2, 3)


        # Run through 2D Conv Stem
        feat_2d = self.stem(x)  # (B, d_model, F_out, T_out)

        # Pool frequency dimension to 1 -> (B, d_model, 1, T)
        pooled = self.freq_pool(feat_2d).squeeze(2)  # (B, d_model, T)
        seq_feat = pooled.transpose(1, 2)  # (B, T, d_model)

        # Dual-Attention
        seq_feat = self.channel_attn(seq_feat)
        seq_feat = self.temporal_attn(seq_feat)

        # Stacked Mamba Blocks
        for block in self.mamba_layers:
            seq_feat = block(seq_feat)

        # Frame-level segmentation logits: (B, T, num_classes)
        seg_logits = self.seg_head(seq_feat)

        # Global sequence classification logits: (B, num_classes)
        global_feat = torch.mean(seq_feat, dim=1)
        cls_logits = self.cls_head(global_feat)

        if return_segmentation:
            return cls_logits, seg_logits
        return cls_logits
