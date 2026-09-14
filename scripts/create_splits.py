"""Leakage-Safe Subject-Independent Train/Validation/Test Split Generator.

Partitions real radar recordings strictly by Subject ID (e.g. P01, P02...)
to guarantee zero identity leakage between training, validation, and testing sets.

Outputs manifest to data/metadata/train_val_test_split.json.
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.adapters.uog_adapter import UoGRadarAdapter


def create_subject_splits(
    raw_dir: str = "data/raw/Dataset_848",
    output_manifest: str = "data/metadata/train_val_test_split.json",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, Any]:
    print("[Split] Discovering samples...")
    adapter = UoGRadarAdapter(root_dir=raw_dir)
    file_list = adapter.discover_samples()

    # Group files by Subject ID
    subject_to_files = defaultdict(list)
    for fpath in file_list:
        meta = adapter.parse_filename_metadata(fpath.name)
        sub_id = meta["subject_id"]
        subject_to_files[sub_id].append({
            "file_path": str(fpath),
            "filename": fpath.name,
            "activity_label": meta["activity_label"],
            "subject_id": sub_id,
            "repetition": meta["repetition"],
        })

    all_subjects = sorted(list(subject_to_files.keys()))
    print(f"[Split] Found {len(all_subjects)} unique subjects across {len(file_list)} files.")

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_subjects = list(all_subjects)
    rng.shuffle(shuffled_subjects)

    n_subjects = len(shuffled_subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    n_test = n_subjects - n_train - n_val

    train_subs = sorted(shuffled_subjects[:n_train])
    val_subs = sorted(shuffled_subjects[n_train : n_train + n_val])
    test_subs = sorted(shuffled_subjects[n_train + n_val :])

    # Build file lists for each partition
    train_files = [f for s in train_subs for f in subject_to_files[s]]
    val_files = [f for s in val_subs for f in subject_to_files[s]]
    test_files = [f for s in test_subs for f in subject_to_files[s]]

    # Ensure zero overlap
    train_sub_set = set(train_subs)
    val_sub_set = set(val_subs)
    test_sub_set = set(test_subs)
    assert train_sub_set.isdisjoint(val_sub_set), "Train and Val subjects overlap!"
    assert train_sub_set.isdisjoint(test_sub_set), "Train and Test subjects overlap!"
    assert val_sub_set.isdisjoint(test_sub_set), "Val and Test subjects overlap!"

    # Count class distribution per partition
    def _count_classes(files: List[Dict[str, Any]]) -> Dict[str, int]:
        c = Counter(f["activity_label"] for f in files)
        return dict(c)

    manifest = {
        "dataset_name": "University of Glasgow Micro-Doppler Radar (Dataset_848)",
        "split_strategy": "Subject-Wise (Leave-Subjects-Out / Leakage-Safe)",
        "random_seed": seed,
        "summary": {
            "total_files": len(file_list),
            "total_subjects": n_subjects,
            "train": {
                "num_subjects": len(train_subs),
                "num_files": len(train_files),
                "subjects": train_subs,
                "class_distribution": _count_classes(train_files),
            },
            "validation": {
                "num_subjects": len(val_subs),
                "num_files": len(val_files),
                "subjects": val_subs,
                "class_distribution": _count_classes(val_files),
            },
            "test": {
                "num_subjects": len(test_subs),
                "num_files": len(test_files),
                "subjects": test_subs,
                "class_distribution": _count_classes(test_files),
            },
        },
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
    }

    out_p = Path(output_manifest)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Split] Manifest created successfully at {output_manifest}!")
    print(f"  Train: {len(train_subs)} subjects, {len(train_files)} files: {_count_classes(train_files)}")
    print(f"  Val:   {len(val_subs)} subjects, {len(val_files)} files: {_count_classes(val_files)}")
    print(f"  Test:  {len(test_subs)} subjects, {len(test_files)} files: {_count_classes(test_files)}")

    return manifest


if __name__ == "__main__":
    create_subject_splits()
