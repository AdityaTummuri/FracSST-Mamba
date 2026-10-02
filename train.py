"""Main training script for FracSST-Mamba radar HAR framework.

Loads dataset and hyperparameters, constructs MambaClassifier sequence network,
runs training & validation loop, computes metrics, and saves model checkpoints.
"""

import argparse
import json
from pathlib import Path
import tempfile
from typing import Dict, Tuple, List, Optional, Any, Union
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW

from data.dataset import create_dataloaders, create_test_dataloader
from models.mamba_block import MambaClassifier
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_detailed_classification_metrics,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration parameters file.

    Args:
        config_path: Path to YAML parameters file.

    Returns:
        Dictionary of parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_batch_inputs(
    specs: torch.Tensor, model_in_features: int
) -> torch.Tensor:
    """Ensure input tensor satisfies the (B, T, F) shape contract.

    Args:
        specs: Input tensor of shape (B, T, F) or (B, 1, F, T) or (B, F, T).
        model_in_features: Expected feature dimension F.

    Returns:
        Tensor reshaped to (B, T, F).
    """
    # 4D case: (B, 1, F, T) -> (B, T, F)
    if specs.ndim == 4 and specs.shape[1] == 1:
        specs = specs.squeeze(1).transpose(1, 2)
    # 3D case where frequency is dim 1: (B, F, T) -> (B, T, F)
    elif (
        specs.ndim == 3
        and specs.shape[1] == model_in_features
        and specs.shape[2] != model_in_features
    ):
        specs = specs.transpose(1, 2)
    return specs


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Execute training epoch over all dataset batches.

    Args:
        model: MambaClassifier model instance.
        train_loader: DataLoader yielding batches of (B, T, F) and (B,).
        optimizer: PyTorch optimizer instance.
        criterion: CrossEntropyLoss criterion.
        device: Target compute device.

    Returns:
        Average training loss across all batches.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0
    in_features = getattr(model, "in_features", 256)

    for batch in train_loader:
        specs, labels = batch[0], batch[1]
        specs = _format_batch_inputs(specs, in_features).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    return running_loss / max(1, total_batches)


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Execute validation evaluation over validation dataset.

    Args:
        model: MambaClassifier model instance.
        val_loader: DataLoader yielding validation batches.
        criterion: CrossEntropyLoss criterion.
        device: Target compute device.

    Returns:
        Tuple containing (val_loss, classification_metrics_dict).
    """
    model.eval()
    running_loss = 0.0
    total_batches = 0
    in_features = getattr(model, "in_features", 256)

    all_cls_true = []
    all_cls_pred = []

    with torch.no_grad():
        for batch in val_loader:
            specs, labels = batch[0], batch[1]
            specs = _format_batch_inputs(specs, in_features).to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            total_batches += 1

            all_cls_true.append(labels.cpu())
            all_cls_pred.append(logits.cpu())

    val_loss = running_loss / max(1, total_batches)
    y_true_cls = (
        torch.cat(all_cls_true, dim=0)
        if all_cls_true
        else torch.tensor([], dtype=torch.long)
    )
    y_pred_cls = (
        torch.cat(all_cls_pred, dim=0)
        if all_cls_pred
        else torch.tensor([], dtype=torch.float32)
    )

    cls_metrics = compute_classification_metrics(y_true_cls, y_pred_cls)
    return val_loss, cls_metrics


def evaluate_test_set(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[Union[str, Path]] = "results",
) -> Dict[str, Any]:
    """Execute complete evaluation over the untouched test partition.

    Args:
        model: MambaClassifier model instance.
        test_loader: DataLoader yielding test batches.
        criterion: CrossEntropyLoss criterion.
        device: Target compute device.
        class_names: Class labels (default: ['Walk', 'Sit', 'Fall']).
        save_dir: Output directory to save evaluation metrics JSON and confusion matrix.

    Returns:
        Dictionary containing test loss, accuracy, macro metrics, per-class metrics, and confusion matrix.
    """
    model.eval()
    if class_names is None:
        class_names = ["Walk", "Sit", "Fall"]

    running_loss = 0.0
    total_batches = 0
    in_features = getattr(model, "in_features", 256)

    all_cls_true = []
    all_cls_pred = []

    with torch.no_grad():
        for batch in test_loader:
            specs, labels = batch[0], batch[1]
            specs = _format_batch_inputs(specs, in_features).to(device)
            labels = labels.to(device)

            logits = model(specs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            total_batches += 1

            all_cls_true.append(labels.cpu())
            all_cls_pred.append(logits.cpu())

    test_loss = running_loss / max(1, total_batches)
    y_true = torch.cat(all_cls_true, dim=0) if all_cls_true else torch.tensor([], dtype=torch.long)
    y_pred = torch.cat(all_cls_pred, dim=0) if all_cls_pred else torch.tensor([], dtype=torch.float32)

    detailed_metrics = compute_detailed_classification_metrics(y_true, y_pred, class_names=class_names)
    detailed_metrics["loss"] = float(test_loss)
    detailed_metrics["total_samples"] = int(len(y_true))

    # Save results if directory specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        results_json_path = save_path / "test_evaluation_results.json"
        serializable_metrics = {
            "loss": float(test_loss),
            "total_samples": int(len(y_true)),
            "accuracy": float(detailed_metrics["accuracy"]),
            "precision_macro": float(detailed_metrics["precision_macro"]),
            "recall_macro": float(detailed_metrics["recall_macro"]),
            "f1_macro": float(detailed_metrics["f1_macro"]),
            "per_class": detailed_metrics["per_class"],
            "confusion_matrix": detailed_metrics["confusion_matrix"].tolist(),
        }
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2)

        try:
            from utils.visualization import plot_confusion_matrix
            cm_plot_path = str(save_path / "test_confusion_matrix.png")
            plot_confusion_matrix(
                cm=detailed_metrics["confusion_matrix"],
                class_names=class_names,
                title="FracSST-Mamba Test Set Confusion Matrix",
                save_path=cm_plot_path,
            )
        except Exception as e:
            print(f"[Warning] Could not plot confusion matrix: {e}")

    return detailed_metrics


def run_smoke_test(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    config: dict,
) -> None:
    """Execute minimal in-memory synthetic smoke test without generating disk files."""
    print("=" * 60)
    print("       FracSST-Mamba: In-Memory Synthetic Smoke Test        ")
    print("=" * 60)
    batch_size = 4
    seq_len = config.get("model", {}).get("seq_len", 16)
    in_features = getattr(model, "in_features", 256)
    num_classes = getattr(model, "num_classes", 3)

    # 1. Synthetic (B, T, F) tensor and (B,) labels
    synthetic_x = torch.randn(batch_size, seq_len, in_features, device=device)
    synthetic_y = torch.randint(0, num_classes, (batch_size,), device=device)

    print(f"[Smoke] Input tensor shape:      {tuple(synthetic_x.shape)} (B, T, F)")
    print(f"[Smoke] Target labels shape:     {tuple(synthetic_y.shape)} (B,)")

    # 2. Forward pass
    optimizer.zero_grad()
    logits = model(synthetic_x)
    print(f"[Smoke] Output logits shape:     {tuple(logits.shape)} (B, {num_classes})")
    assert logits.shape == (batch_size, num_classes), (
        f"Expected shape {(batch_size, num_classes)}, got {logits.shape}"
    )

    # 3. CrossEntropyLoss
    loss = criterion(logits, synthetic_y)
    print(f"[Smoke] CrossEntropyLoss:        {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss evaluated to NaN"

    # 4. Backward gradient propagation
    loss.backward()
    print("[Smoke] Backward gradient pass:  PASS")

    # 5. Optimizer step
    optimizer.step()
    print("[Smoke] Optimizer update step:   PASS")

    # 6. Checkpoint save & load verification using isolated temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        ckpt_path = Path(temp_dir) / "smoke_checkpoint.pth"
        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            epoch=1,
            val_accuracy=1.0,
            config=config,
        )
        assert ckpt_path.exists(), "Checkpoint file was not created"
        ckpt_data = load_checkpoint(
            ckpt_path, model=model, optimizer=optimizer, device=device
        )
        assert ckpt_data.get("epoch") == 1, "Restored checkpoint epoch mismatch"
        print("[Smoke] Checkpoint save & load:  PASS")

    print("=" * 60)
    print("Synthetic smoke test completed successfully!")
    print("Pipeline contract (B, T, F) -> MambaClassifier -> (B, 3) is verified.")
    print("=" * 60)


def main() -> None:
    """Run model training and evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Train FracSST-Mamba sequence classifier for Radar Human Activity Recognition."
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run minimal in-memory synthetic smoke test without disk data.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file (.pth) to resume training from.",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Path to checkpoint file (.pth) to evaluate without training.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default="data/metadata/train_val_test_split.json",
        help="Path to subject-independent train/val/test split manifest JSON.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate model checkpoint on the untouched test partition.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="weights/best_model.pth",
        help="Path to model checkpoint (.pth) to load for evaluation (default: weights/best_model.pth).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[Train] Loaded configuration parameters from {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Target execution compute device: {device}")

    # Hyperparameters
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 32)
    n_epochs = 1 if args.dry_run else (args.epochs or config.get("training", {}).get("epochs", 50))
    learning_rate = config.get("training", {}).get("lr", 0.001)

    model_cfg = config.get("model", {})
    in_features = model_cfg.get("in_features", 256)
    d_model = model_cfg.get("d_model", 128)
    n_layers = model_cfg.get("n_layers", 4)
    dropout = model_cfg.get("dropout", 0.1)
    num_classes = model_cfg.get("num_classes", 3)

    # Initialize MambaClassifier (Target Architecture)
    model = MambaClassifier(
        in_features=in_features,
        d_model=d_model,
        n_layers=n_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] MambaClassifier initialized with {total_params:,} trainable parameters.")

    # Optimizer & Single-Task Classification Loss
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()

    # In-memory smoke test execution path
    if args.smoke_test:
        run_smoke_test(
            model=model,
            optimizer=optimizer,
            criterion=criterion_cls,
            device=device,
            config=config,
        )
        return

    # Check for preprocessed dataset
    processed_dir = Path("data/processed")
    processed_files = list(processed_dir.glob("*.pt")) if processed_dir.exists() else []
    if len(processed_files) == 0:
        print("[Train] No preprocessed radar tensors found in data/processed/.")
        print("[Train] Real dataset ingestion and preprocessing is in progress by the dataset teammate.")
        print("[Train] To verify the full training pipeline in-memory right now, run:")
        print("        python train.py --smoke-test")
        return

    # DataLoaders
    split_manifest_path = Path(args.split_manifest) if args.split_manifest else None
    if split_manifest_path and split_manifest_path.exists():
        print(f"[Train] Using subject-independent split from: {split_manifest_path}")

    train_loader, val_loader = create_dataloaders(
        data_dir=processed_dir,
        batch_size=batch_size,
        val_split=0.2,
        split_manifest=split_manifest_path,
    )
    print(
        f"[Train] Loaded {len(train_loader.dataset)} training samples, "
        f"{len(val_loader.dataset)} validation samples."
    )

    start_epoch = 1
    best_val_acc = -1.0

    # Resume or Eval-Only Checkpoint Restore
    ckpt_path = args.eval_only or args.resume
    if ckpt_path:
        ckpt_data = load_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer if args.resume else None,
            device=device,
        )
        print(f"[Train] Loaded checkpoint from: {ckpt_path}")
        if "val_accuracy" in ckpt_data:
            best_val_acc = ckpt_data["val_accuracy"]
            print(f"[Train] Restored checkpoint validation accuracy: {best_val_acc * 100:.2f}%")
        if args.resume and "epoch" in ckpt_data:
            start_epoch = ckpt_data["epoch"] + 1
            print(f"[Train] Resuming training from epoch {start_epoch}")

    # Test Evaluation Execution Path (Untouched Test Partition)
    if args.test:
        test_ckpt = args.checkpoint or args.eval_only or "weights/best_model.pth"
        print("==================================================")
        print("  Evaluating FracSST-Mamba on Untouched Test Set  ")
        print("==================================================")
        ckpt_data = load_checkpoint(
            checkpoint_path=test_ckpt,
            model=model,
            device=device,
        )
        print(f"[Test] Loaded checkpoint from: {test_ckpt}")
        if "epoch" in ckpt_data:
            print(f"[Test] Trained Epochs: {ckpt_data['epoch']}")
        if "val_accuracy" in ckpt_data:
            print(f"[Test] Restored Checkpoint Val Accuracy: {ckpt_data['val_accuracy'] * 100:.2f}%")

        split_manifest_path = Path(args.split_manifest) if args.split_manifest else Path("data/metadata/train_val_test_split.json")
        test_loader = create_test_dataloader(
            data_dir=processed_dir,
            split_manifest=split_manifest_path,
            batch_size=batch_size,
        )
        print(f"[Test] Loaded {len(test_loader.dataset)} untouched test samples from {split_manifest_path}")

        # Strict subject-independent leakage check
        with open(split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        train_subs = set(manifest["summary"]["train"]["subjects"])
        val_subs = set(manifest["summary"]["validation"]["subjects"])
        test_subs = set(manifest["summary"]["test"]["subjects"])
        assert train_subs.isdisjoint(test_subs), "Subject leakage detected between train and test sets!"
        assert val_subs.isdisjoint(test_subs), "Subject leakage detected between validation and test sets!"
        print(f"[Test] Leakage check passed: Zero subject overlap with Train ({len(train_subs)} subjects) or Val ({len(val_subs)} subjects).")
        print(f"[Test] Test partition spans {len(test_subs)} unique subjects: {sorted(list(test_subs))}")

        test_results = evaluate_test_set(
            model=model,
            test_loader=test_loader,
            criterion=criterion_cls,
            device=device,
            class_names=["Walk", "Sit", "Fall"],
            save_dir="results",
        )

        print("--------------------------------------------------")
        print(f"Test Loss:             {test_results['loss']:.4f}")
        print(f"Test Accuracy:         {test_results['accuracy'] * 100:.2f}%")
        print(f"Test Macro F1:         {test_results['f1_macro']:.4f}")
        print(f"Test Macro Precision:  {test_results['precision_macro']:.4f}")
        print(f"Test Macro Recall:     {test_results['recall_macro']:.4f}")
        print("--------------------------------------------------")
        print("Per-Class Metrics:")
        for c_name, c_metrics in test_results["per_class"].items():
            print(
                f"  {c_name:<6} | Prec: {c_metrics['precision'] * 100:.2f}% | "
                f"Rec: {c_metrics['recall'] * 100:.2f}% | "
                f"F1: {c_metrics['f1']:.4f} | Support: {c_metrics['support']}"
            )
        print("--------------------------------------------------")
        print("Confusion Matrix (Rows: True, Cols: Predicted):")
        cm = test_results["confusion_matrix"]
        print(f"{'':>8} {'Walk':>8} {'Sit':>8} {'Fall':>8}")
        for i, c_name in enumerate(["Walk", "Sit", "Fall"]):
            row_str = " ".join(f"{cm[i, j]:>8d}" for j in range(3))
            print(f"{c_name:>8} {row_str}")
        print("==================================================")
        print("Results saved to: results/test_evaluation_results.json and results/test_confusion_matrix.png")
        print("==================================================")
        return

    # Eval-Only Execution Path
    if args.eval_only:
        print("==================================================")
        print("     Evaluating Model Checkpoints (Eval-Only)     ")
        print("==================================================")
        val_loss, cls_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion_cls,
            device=device,
        )
        print(f"Validation Loss:             {val_loss:.4f}")
        print(f"Classification Accuracy:     {cls_metrics['accuracy'] * 100:.2f}%")
        print(f"Classification Macro F1:     {cls_metrics['f1_macro']:.4f}")
        print(f"Classification Macro Prec:   {cls_metrics['precision_macro']:.4f}")
        print(f"Classification Macro Recall: {cls_metrics['recall_macro']:.4f}")
        print("==================================================")
        return

    checkpoint_dir = Path(config.get("training", {}).get("checkpoint_dir", "weights"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "best_model.pth"

    print("==================================================")
    print("      Starting FracSST-Mamba Model Training       ")
    print("==================================================")

    for epoch in range(start_epoch, start_epoch + n_epochs):
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion_cls,
            device=device,
        )

        val_loss, cls_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion_cls,
            device=device,
        )

        val_acc = cls_metrics["accuracy"]
        val_f1 = cls_metrics["f1_macro"]

        print(
            f"Epoch [{epoch:02d}/{start_epoch + n_epochs - 1:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        # Checkpoint saving when validation accuracy improves
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                checkpoint_path=best_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_acc,
                config=config,
            )
            print(
                f"  --> Saved new best checkpoint to {best_model_path} (Val Acc: {val_acc * 100:.2f}%)"
            )

    print("==================================================")
    print(f"Training Complete! Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best model weights saved to: {best_model_path}")
    print("==================================================")



if __name__ == "__main__":
    main()
