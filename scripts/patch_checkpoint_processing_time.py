"""
Patch checkpoint files with processing_time from pre-repair backup files.

For each dataset's checkpoint file, finds the corresponding pre-repair backup,
reads processing_time values keyed by record index, and writes them into
the checkpoint records (only for non-None records that don't already have
processing_time).

Usage:
    python scripts/patch_checkpoint_processing_time.py [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"
MODEL_NAME = "MiniMax-M2.7"

# Dataset -> (split,) mapping
DATASETS = {
    "aime": "hybrid",
    "arcc": "test",
    "bbh": "test",
    "emorynlp": "test",
    "finqa": "test",
    "gpqa": "test",
    "livecodebench": "test",
    "livemathbench": "test",
    "math500": "test",
    "mbpp": "test",
    "medqa": "test",
    "meld": "test",
    "mmlupro": "test_3000",
    "winogrande": "valid",
}


def find_pre_repair(model_dir: Path) -> Path | None:
    """Find the pre-repair backup file in the given directory."""
    candidates = sorted(model_dir.glob("*-pre-repair.json"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  WARNING: Multiple pre-repair files in {model_dir}, using latest")
        return candidates[-1]
    return None


def patch_dataset(dataset_id: str, split: str, dry_run: bool) -> dict:
    """Patch a single dataset's checkpoint with processing_time from pre-repair.

    Returns a stats dict with counts.
    """
    model_dir = RESULTS_DIR / dataset_id / split / MODEL_NAME
    checkpoint_path = model_dir / f"{dataset_id}-{split}-{MODEL_NAME}-checkpoint.json"

    stats = {"dataset": dataset_id, "patched": 0, "skipped_none": 0, "skipped_has_pt": 0, "missing_in_backup": 0, "total": 0}

    if not checkpoint_path.exists():
        print(f"  SKIP: No checkpoint file: {checkpoint_path.name}")
        return stats

    pre_repair_path = find_pre_repair(model_dir)
    if pre_repair_path is None:
        print(f"  SKIP: No pre-repair backup in {model_dir}")
        return stats

    # Load pre-repair and build index -> processing_time map
    with open(pre_repair_path, "r", encoding="utf-8") as f:
        pre_repair_data = json.load(f)

    pt_map: dict[int, float] = {}
    for rec in pre_repair_data.get("records", []):
        idx = rec.get("index")
        pt = rec.get("processing_time")
        if idx is not None and pt is not None:
            pt_map[idx] = pt

    # Load checkpoint
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        cp_data = json.load(f)

    records = cp_data.get("records", [])
    stats["total"] = len(records)

    modified = False
    for i, rec in enumerate(records):
        if rec is None:
            stats["skipped_none"] += 1
            continue

        if "processing_time" in rec and rec["processing_time"] != 0.0:
            stats["skipped_has_pt"] += 1
            continue

        rec_index = rec.get("index", i)
        if rec_index in pt_map:
            rec["processing_time"] = pt_map[rec_index]
            stats["patched"] += 1
            modified = True
        else:
            stats["missing_in_backup"] += 1

    if modified and not dry_run:
        tmp_path = checkpoint_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cp_data, f, ensure_ascii=False)
        os.replace(str(tmp_path), str(checkpoint_path))

    return stats


def main():
    parser = argparse.ArgumentParser(description="Patch checkpoint files with processing_time from pre-repair backups")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing files")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===\n")

    total_patched = 0
    total_missing = 0

    for dataset_id, split in DATASETS.items():
        print(f"[{dataset_id}/{split}]")
        stats = patch_dataset(dataset_id, split, args.dry_run)

        if stats["patched"] > 0 or stats["missing_in_backup"] > 0:
            print(f"  Records: {stats['total']} total, {stats['patched']} patched, "
                  f"{stats['skipped_none']} null, {stats['skipped_has_pt']} already had pt, "
                  f"{stats['missing_in_backup']} missing in backup")
        elif stats["total"] > 0:
            print(f"  Records: {stats['total']} total - no changes needed")

        total_patched += stats["patched"]
        total_missing += stats["missing_in_backup"]

    print(f"\n{'DRY RUN ' if args.dry_run else ''}SUMMARY: {total_patched} records patched, {total_missing} missing from backup")


if __name__ == "__main__":
    main()
