"""Verification script for Stage 2 neural network architectures and utilities.

Instantiates MambaBlock, DASegModel, feeds dummy radar spectrogram tensors,
computes classification & segmentation loss gradients, and tests evaluation metrics.
"""

import sys
import torch
import torch.nn as nn
import numpy as np

from models.mamba_block import MambaBlock
from models.da_seg import DASegModel
from utils.metrics import compute_classification_metrics, compute_segmentation_metrics


def main() -> None:
    print("==================================================")
    print("    FracSST-Mamba Stage 2 Neural Net Verification ")
    print("==================================================")

    device = torch.device("cpu")
    batch_size = 4
    channels = 1
    freq_bins = 256
    time_steps = 16
    d_model = 128
    num_classes = 3

    print("[1] Testing MambaBlock sequence module...")
    dummy_seq = torch.randn(batch_size, time_steps, d_model, device=device)
    mamba_block = MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2, dropout=0.1)
    out_seq = mamba_block(dummy_seq)

    assert out_seq.shape == dummy_seq.shape, f"MambaBlock shape mismatch: {out_seq.shape} vs {dummy_seq.shape}"
    print(f"    - MambaBlock Input Shape:  {dummy_seq.shape}")
    print(f"    - MambaBlock Output Shape: {out_seq.shape}")

    print("\n[2] Testing DASegModel Architecture...")
    dummy_spec = torch.randn(batch_size, channels, freq_bins, time_steps, device=device)
    model = DASegModel(
        in_channels=channels,
        d_model=d_model,
        n_layers=2,
        num_classes=num_classes,
        dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - DASegModel Total Trainable Parameters: {param_count:,}")

    cls_logits, seg_logits = model(dummy_spec, return_segmentation=True)

    assert cls_logits.shape == (batch_size, num_classes), f"CLS shape error: {cls_logits.shape}"
    assert seg_logits.shape == (batch_size, time_steps, num_classes), f"SEG shape error: {seg_logits.shape}"

    print(f"    - Input Spectrogram Shape: {dummy_spec.shape}")
    print(f"    - Classification Logits:  {cls_logits.shape}")
    print(f"    - Segmentation Logits:    {seg_logits.shape}")

    print("\n[3] Testing Loss Computation & Backward Pass...")
    target_cls = torch.randint(0, num_classes, (batch_size,), device=device)
    target_seg = torch.randint(0, num_classes, (batch_size, time_steps), device=device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    loss_cls = criterion_cls(cls_logits, target_cls)
    loss_seg = criterion_seg(seg_logits.view(-1, num_classes), target_seg.view(-1))
    total_loss = loss_cls + loss_seg

    total_loss.backward()
    print(f"    - Combined Loss (CLS + SEG): {total_loss.item():.4f}")
    print("    - Backward gradient pass successful!")

    print("\n[4] Testing Evaluation Metrics...")
    cls_metrics = compute_classification_metrics(target_cls, cls_logits)
    seg_metrics = compute_segmentation_metrics(target_seg, seg_logits)

    print(f"    - Classification Metrics: {cls_metrics}")
    print(f"    - Segmentation Metrics:   {seg_metrics}")

    print("\n==================================================")
    print("   Stage 2 Architecture Executed Successfully!   ")
    print("==================================================")


if __name__ == "__main__":
    main()
