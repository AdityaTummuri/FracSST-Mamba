"""Main training script for FracSST-Mamba radar HAR framework.

Loads dataset and hyperparameters, constructs DSP pipeline and FracSST-Mamba network,
runs training & validation loop, computes metrics, and saves best model checkpoints.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

from data.dataset import create_dataloaders
from dsp.synchrosqueezing import fracsst_1d
from models.da_seg import DASegModel
from preprocess import preprocess_all
from utils.metrics import compute_classification_metrics, compute_segmentation_metrics


def load_config(config_path: str) -> dict:
    """Load YAML configuration parameters file.

    Args:
        config_path: Path to YAML parameters file.

    Returns:
        Dictionary of parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_cls: nn.Module,
    criterion_seg: nn.Module,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Execute training epoch over all dataset batches.

    Returns:
        Tuple containing (total_loss, class_loss, seg_loss).
    """
    model.train()
    running_loss = 0.0
    running_cls_loss = 0.0
    running_seg_loss = 0.0
    total_batches = 0

    for specs, labels, seg_masks in train_loader:
        specs = specs.to(device)
        labels = labels.to(device)
        seg_masks = seg_masks.to(device)

        optimizer.zero_grad()

        cls_logits, seg_logits = model(specs, return_segmentation=True)

        loss_cls = criterion_cls(cls_logits, labels)
        loss_seg = criterion_seg(
            seg_logits.view(-1, num_classes), seg_masks.view(-1)
        )
        loss = loss_cls + loss_seg

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_cls_loss += loss_cls.item()
        running_seg_loss += loss_seg.item()
        total_batches += 1

    avg_loss = running_loss / max(1, total_batches)
    avg_cls_loss = running_cls_loss / max(1, total_batches)
    avg_seg_loss = running_seg_loss / max(1, total_batches)

    return avg_loss, avg_cls_loss, avg_seg_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion_cls: nn.Module,
    criterion_seg: nn.Module,
    num_classes: int,
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Execute validation evaluation over validation dataset.

    Returns:
        Tuple containing (val_loss, classification_metrics_dict, segmentation_metrics_dict).
    """
    model.eval()
    running_loss = 0.0
    total_batches = 0

    all_cls_true = []
    all_cls_pred = []
    all_seg_true = []
    all_seg_pred = []

    with torch.no_grad():
        for specs, labels, seg_masks in val_loader:
            specs = specs.to(device)
            labels = labels.to(device)
            seg_masks = seg_masks.to(device)

            cls_logits, seg_logits = model(specs, return_segmentation=True)

            loss_cls = criterion_cls(cls_logits, labels)
            loss_seg = criterion_seg(
                seg_logits.view(-1, num_classes), seg_masks.view(-1)
            )
            loss = loss_cls + loss_seg

            running_loss += loss.item()
            total_batches += 1

            all_cls_true.append(labels.cpu())
            all_cls_pred.append(cls_logits.cpu())
            all_seg_true.append(seg_masks.cpu())
            all_seg_pred.append(seg_logits.cpu())

    val_loss = running_loss / max(1, total_batches)

    # Concatenate predictions across batches
    y_true_cls = torch.cat(all_cls_true, dim=0)
    y_pred_cls = torch.cat(all_cls_pred, dim=0)
    y_true_seg = torch.cat(all_seg_true, dim=0)
    y_pred_seg = torch.cat(all_seg_pred, dim=0)

    cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
    seg_metrics = compute_segmentation_metrics(y_true_seg, y_pred_seg)

    return val_loss, cls_metrics, seg_metrics


def main() -> None:
    """Run model training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train FracSST-Mamba model for Radar Human Activity Recognition."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/params.yaml",
        help="Path to YAML configuration parameters file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 epoch dry-run for pipeline verification.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Train] Loaded configuration parameters from {args.config}")

    # Ensure preprocessed data exists
    processed_dir = Path("data/processed")
    if not processed_dir.exists() or len(list(processed_dir.glob("*.pt"))) == 0:
        print("[Train] Preprocessed data not found. Running offline DSP preprocessor...")
        preprocess_all(Path("data/raw"), processed_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Target execution compute device: {device}")

    # Hyperparameters
    batch_size = args.batch_size or config["training"].get("batch_size", 32)
    n_epochs = 1 if args.dry_run else (args.epochs or config["training"].get("epochs", 50))
    learning_rate = config["training"].get("lr", 0.001)

    model_cfg = config.get("model", {})
    d_model = model_cfg.get("d_model", 128)
    n_layers = model_cfg.get("n_layers", 4)
    dropout = model_cfg.get("dropout", 0.1)
    num_classes = model_cfg.get("num_classes", 3)

    # Initialize Model
    model = DASegModel(
        in_channels=1,
        d_model=d_model,
        n_layers=n_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] DASegModel initialized with {total_params:,} trainable parameters.")

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        data_dir=processed_dir,
        batch_size=batch_size,
        val_split=0.2,
    )
    print(
        f"[Train] Loaded {len(train_loader.dataset)} training samples, "
        f"{len(val_loader.dataset)} validation samples."
    )

    # Optimizer & Losses
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = weights_dir / "best_model.pth"

    best_val_acc = -1.0

    print("==================================================")
    print("      Starting FracSST-Mamba Model Training       ")
    print("==================================================")

    for epoch in range(1, n_epochs + 1):
        train_loss, train_cls_loss, train_seg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_cls=criterion_cls,
            criterion_seg=criterion_seg,
            num_classes=num_classes,
            device=device,
        )

        val_loss, cls_metrics, seg_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion_cls=criterion_cls,
            criterion_seg=criterion_seg,
            num_classes=num_classes,
            device=device,
        )

        val_acc = cls_metrics["accuracy"]
        val_miou = seg_metrics["mIoU"]
        val_frame_acc = seg_metrics["frame_accuracy"]

        print(
            f"Epoch [{epoch:02d}/{n_epochs:02d}] | "
            f"Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Seg: {train_seg_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Frame Acc: {val_frame_acc * 100:.2f}% | "
            f"Val mIoU: {val_miou:.4f}"
        )

        # Checkpoint saving when validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_mIoU": val_miou,
                    "config": config,
                },
                best_model_path,
            )
            print(f"  --> Saved new best checkpoint to {best_model_path} (Val Acc: {val_acc * 100:.2f}%)")

    print("==================================================")
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best model weights saved to: {best_model_path}")
    print("==================================================")


if __name__ == "__main__":
    main()
