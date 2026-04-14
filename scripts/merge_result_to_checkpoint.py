"""
Merge new result files with pre-repair backups to create checkpoints.

For datasets where repair runs completed but still have failures:
1. Identifies records recovered in the new run (was failed, now success)
2. Identifies records still failing (marks as None for retry)
3. For records that were always successful, restores processing_time from pre-repair
4. Creates a checkpoint file and optionally deletes the new result file

Usage:
    python scripts/merge_result_to_checkpoint.py [--dry-run] [--delete-result]
"""

import argparse
import json
import os
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "bench"
MODEL_NAME = "MiniMax-M2.7"

# 12 fields matching runner.py _save_checkpoint
CHECKPOINT_FIELDS = [
    "index", "origin_query", "prompt", "prompt_tokens", "completion_tokens",
    "cost", "score", "prediction", "ground_truth", "raw_output",
    "processing_time", "extra_fields",
]

# Datasets to process: (dataset_id, split, new_result_filename)
TARGETS = [
    ("aime", "hybrid", "aime-hybrid-MiniMax-M2.7-20260408_091522.json"),
    ("gpqa", "test", "gpqa-test-MiniMax-M2.7-20260408_163929.json"),
    ("livemathbench", "test", "livemathbench-test-MiniMax-M2.7-20260408_113251.json"),
    ("math500", "test", "math500-test-MiniMax-M2.7-20260408_124933.json"),
    ("mbpp", "test", "mbpp-test-MiniMax-M2.7-20260409_010117.json"),
]


def is_failed(record: dict) -> bool:
    raw = record.get("raw_output", "")
    return isinstance(raw, str) and (
        raw.startswith("Generation failed") or raw.startswith("Processing failed")
    )


def to_checkpoint_record(record: dict) -> dict:
    return {k: record.get(k) for k in CHECKPOINT_FIELDS}


def find_pre_repair(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("*-pre-repair.json"))
    return candidates[-1] if candidates else None


def process_dataset(dataset_id: str, split: str, result_filename: str, dry_run: bool, delete_result: bool) -> dict:
    model_dir = RESULTS_DIR / dataset_id / split / MODEL_NAME
    new_path = model_dir / result_filename
    pre_path = find_pre_repair(model_dir)

    stats = {"dataset": dataset_id, "total": 0, "recovered": 0, "still_failed": 0,
             "preserved": 0, "pt_fixed": 0, "status": ""}

    if not new_path.exists():
        stats["status"] = f"SKIP: new result not found: {result_filename}"
        return stats

    if pre_path is None:
        stats["status"] = "SKIP: no pre-repair backup"
        return stats

    # Load both files
    new_data = json.load(open(new_path, "r", encoding="utf-8"))
    pre_data = json.load(open(pre_path, "r", encoding="utf-8"))

    new_recs = new_data["records"]
    pre_recs = pre_data["records"]
    total = len(new_recs)
    stats["total"] = total

    # Build pre-repair index -> record map
    pre_map = {r["index"]: r for r in pre_recs}
    pre_failed_idx = {r["index"] for r in pre_recs if is_failed(r)}

    # Build checkpoint records
    checkpoint_records = []
    for new_rec in new_recs:
        idx = new_rec["index"]
        pre_rec = pre_map.get(idx)

        if is_failed(new_rec):
            # Still failing -> None (retry later)
            checkpoint_records.append(None)
            stats["still_failed"] += 1
        elif idx in pre_failed_idx:
            # Recovered! Was failed in pre-repair, now success
            # Use new record as-is (has correct pt from the new run)
            checkpoint_records.append(to_checkpoint_record(new_rec))
            stats["recovered"] += 1
        else:
            # Always was successful
            # Use new record but fix processing_time from pre-repair
            merged = to_checkpoint_record(new_rec)
            if pre_rec and pre_rec.get("processing_time", 0) > 0:
                merged["processing_time"] = pre_rec["processing_time"]
                stats["pt_fixed"] += 1
            stats["preserved"] += 1
            checkpoint_records.append(merged)

    completed = sum(1 for r in checkpoint_records if r is not None)

    checkpoint_payload = {
        "checkpoint": True,
        "completed_count": completed,
        "total_count": total,
        "dataset_id": dataset_id,
        "split": split,
        "model_name": MODEL_NAME,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": checkpoint_records,
    }

    cp_path = model_dir / f"{dataset_id}-{split}-{MODEL_NAME}-checkpoint.json"

    if not dry_run:
        tmp = cp_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(checkpoint_payload, f, ensure_ascii=False)
        os.replace(str(tmp), str(cp_path))

        if delete_result:
            os.remove(new_path)
            stats["status"] = f"checkpoint created, result deleted"
        else:
            stats["status"] = f"checkpoint created, result kept"
    else:
        stats["status"] = "would create checkpoint"
        if delete_result:
            stats["status"] += " + delete result"

    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge new results with pre-repair into checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing files")
    parser.add_argument("--delete-result", action="store_true", help="Delete new result files after creating checkpoints")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN ===\n")

    for dataset_id, split, filename in TARGETS:
        print(f"[{dataset_id}/{split}]")
        stats = process_dataset(dataset_id, split, filename, args.dry_run, args.delete_result)
        print(f"  total={stats['total']}, recovered={stats['recovered']}, "
              f"still_failed={stats['still_failed']}, preserved={stats['preserved']}, "
              f"pt_fixed={stats['pt_fixed']}")
        print(f"  -> {stats['status']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
