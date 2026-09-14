"""Utility script to prune unnecessary radar recordings and reclaim disk space.

Retains only the 3 project target classes:
- Prefix '1': Walking (Project Class 0)
- Prefix '2': Sitting down (Project Class 1)
- Prefix '6': Simulated Fall (Project Class 2)

Removes:
- Unused activities: '3' (Stand up), '4' (Pick up object), '5' (Drink water)
- Campaign 6 (NG Homes) and Campaign 7 (West Cumbria) which lack fall events.
"""

import shutil
from pathlib import Path

RAW_DIR = Path("data/raw/Dataset_848")
EXCLUDED_CAMPAIGNS = [
    "6 February 2019 NG Homes Dataset",
    "7 March 2019 West Cumbria Dataset",
]
TARGET_PREFIXES = {"1", "2", "6"}


def prune_dataset(dry_run: bool = False) -> None:
    if not RAW_DIR.exists():
        print(f"Error: Raw directory {RAW_DIR} does not exist.")
        return

    total_deleted_files = 0
    total_freed_bytes = 0

    print(f"[Prune] Starting data cleanup in {RAW_DIR} (dry_run={dry_run})...")

    # 1. Remove entire campaigns that lack falls
    for campaign_name in EXCLUDED_CAMPAIGNS:
        campaign_path = RAW_DIR / campaign_name
        if campaign_path.exists():
            folder_size = sum(f.stat().st_size for f in campaign_path.rglob("*") if f.is_file())
            file_count = sum(1 for f in campaign_path.rglob("*") if f.is_file())
            print(f"[Prune] Removing campaign '{campaign_name}' ({file_count} files, {folder_size / 1e9:.2f} GB)...")
            if not dry_run:
                shutil.rmtree(campaign_path)
            total_deleted_files += file_count
            total_freed_bytes += folder_size

    # 2. In retained campaigns, remove non-target activity files
    retained_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    for folder in sorted(retained_dirs):
        for dat_file in sorted(folder.glob("*.dat")):
            prefix = dat_file.name[0]
            if prefix not in TARGET_PREFIXES:
                file_size = dat_file.stat().st_size
                total_deleted_files += 1
                total_freed_bytes += file_size
                if not dry_run:
                    dat_file.unlink()

    print(f"\n[Prune] Complete! Deleted {total_deleted_files} files.")
    print(f"[Prune] Freed: {total_freed_bytes / (1024**3):.2f} GB of disk space.")


if __name__ == "__main__":
    prune_dataset(dry_run=False)
