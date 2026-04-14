#!/usr/bin/env python3
"""
Convert MiniMax-M2.7 result JSON files to checkpoint format for repair runs.

This script scans results/bench/ for MiniMax-M2.7 result files, identifies
records with "Generation failed" in raw_output, renames the original result
file to *-pre-repair.json, and creates a checkpoint file that marks failed
records as None for re-processing.

Usage:
    python scripts/result_to_checkpoint.py [--dry-run]

Arguments:
    --dry-run: Print statistics only, don't modify any files
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Checkpoint format: 12 fields per record (matches runner.py _save_checkpoint)
CHECKPOINT_RECORD_FIELDS = [
    "index",
    "origin_query",
    "prompt",
    "prompt_tokens",
    "completion_tokens",
    "cost",
    "score",
    "prediction",
    "ground_truth",
    "raw_output",
    "processing_time",
    "extra_fields",
]


def is_failed_record(record: Dict[str, Any]) -> bool:
    """Check if a record has Generation failed or Processing failed."""
    raw_output = record.get("raw_output", "")
    if not isinstance(raw_output, str):
        return False
    return raw_output.startswith("Generation failed") or raw_output.startswith("Processing failed")


def convert_record_to_checkpoint_format(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the 11 checkpoint fields from a result record."""
    return {field: record.get(field) for field in CHECKPOINT_RECORD_FIELDS}


def parse_result_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse result filename to extract dataset, split, model.
    Expected format: {dataset}-{split}-{model}-{timestamp}.json
    """
    # Match pattern like: emorynlp-test-MiniMax-M2.7-20260403_212348.json
    pattern = r"^(.+?)-(.+?)-(MiniMax-M2\.7)-(\d{8}_\d{6})\.json$"
    match = re.match(pattern, filename)
    if match:
        return {
            "dataset_id": match.group(1),
            "split": match.group(2),
            "model_name": match.group(3),
            "timestamp": match.group(4),
        }
    return None


def find_result_files(bench_dir: Path) -> List[Path]:
    """
    Find all MiniMax-M2.7 result files in results/bench/.
    Excludes *-pre-repair.json and *-checkpoint.json files.
    """
    result_files = []
    for result_file in bench_dir.rglob("MiniMax-M2.7/*-20*.json"):
        if "pre-repair" in result_file.name or "checkpoint" in result_file.name:
            continue
        result_files.append(result_file)
    return sorted(result_files)


def process_result_file(
    result_file: Path, dry_run: bool = False, force: bool = False
) -> Dict[str, Any]:
    """
    Process a single result file:
    1. Read and count failed records
    2. If failures exist and not dry-run:
       - Rename result file to *-pre-repair.json
       - Create checkpoint file with failed records as None
    
    Returns statistics dict with total/failed/preserved counts.
    """
    # Read result file
    with open(result_file, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    
    records = result_data.get("records", [])
    total_count = len(records)
    failed_indices = [i for i, r in enumerate(records) if is_failed_record(r)]
    failed_count = len(failed_indices)
    preserved_count = total_count - failed_count
    
    # Parse filename
    parsed = parse_result_filename(result_file.name)
    if not parsed:
        return {
            "file": result_file.name,
            "total": total_count,
            "failed": failed_count,
            "preserved": preserved_count,
            "status": "skipped (unparsable filename)",
        }
    
    dataset_id = parsed["dataset_id"]
    split = parsed["split"]
    model_name = parsed["model_name"]
    
    # Skip if no failures
    if failed_count == 0:
        return {
            "dataset": dataset_id,
            "total": total_count,
            "failed": 0,
            "preserved": preserved_count,
            "status": "no failures",
        }
    
    # Check if already processed (pre-repair file exists)
    pre_repair_file = result_file.with_name(
        result_file.name.replace(".json", "-pre-repair.json")
    )
    if pre_repair_file.exists() and not force:
        return {
            "dataset": dataset_id,
            "total": total_count,
            "failed": failed_count,
            "preserved": preserved_count,
            "status": "already processed (pre-repair exists)",
        }
    
    # If force mode and pre-repair exists, use pre-repair as source
    if force and pre_repair_file.exists():
        print(f"Force mode: using pre-repair file as source for {dataset_id}")
        source_file = pre_repair_file
        # Don't rename again - checkpoint will be created from pre-repair
        skip_rename = True
    else:
        source_file = result_file
        skip_rename = False
    
    if dry_run:
        return {
            "dataset": dataset_id,
            "total": total_count,
            "failed": failed_count,
            "preserved": preserved_count,
            "status": "would convert (force)" if force else "would convert",
        }
    
    # Rename result file to -pre-repair.json (unless using existing pre-repair)
    if not skip_rename:
        os.rename(result_file, pre_repair_file)
    
    # Build checkpoint payload
    checkpoint_records = []
    for i, record in enumerate(records):
        if is_failed_record(record):
            checkpoint_records.append(None)
        else:
            checkpoint_records.append(convert_record_to_checkpoint_format(record))
    
    checkpoint_payload = {
        "checkpoint": True,
        "completed_count": preserved_count,
        "total_count": total_count,
        "dataset_id": dataset_id,
        "split": split,
        "model_name": model_name,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "records": checkpoint_records,
    }
    
    # Write checkpoint atomically
    checkpoint_file = result_file.with_name(
        f"{dataset_id}-{split}-{model_name}-checkpoint.json"
    )
    tmp_file = checkpoint_file.with_suffix(".tmp")
    
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_payload, f, ensure_ascii=False, indent=2)
    
    os.replace(tmp_file, checkpoint_file)
    
    return {
        "dataset": dataset_id,
        "total": total_count,
        "failed": failed_count,
        "preserved": preserved_count,
        "status": "converted",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert MiniMax-M2.7 result files to checkpoint format"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics only, don't modify files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate checkpoints even if pre-repair files exist",
    )
    args = parser.parse_args()
    
    # Locate results/bench directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    bench_dir = project_root / "results" / "bench"
    
    if not bench_dir.exists():
        print(f"Error: results/bench directory not found at {bench_dir}")
        return 1
    
    # Find all MiniMax-M2.7 result files
    result_files = find_result_files(bench_dir)
    
    if not result_files:
        print("No MiniMax-M2.7 result files found.")
        return 0
    
    print(f"{'='*80}")
    print(f"MiniMax-M2.7 Result → Checkpoint Conversion")
    print(f"{'='*80}")
    mode_desc = "DRY-RUN (no file changes)" if args.dry_run else "LIVE (will modify files)"
    if args.force:
        mode_desc += " [FORCE MODE]"
    print(f"Mode: {mode_desc}")
    print(f"Found {len(result_files)} result files\n")
    
    # Process each file
    stats_list = []
    for result_file in result_files:
        stats = process_result_file(result_file, dry_run=args.dry_run, force=args.force)
        stats_list.append(stats)
    
    # Print summary table
    print(f"\n{'Dataset':<20} {'Total':<8} {'Failed':<8} {'Preserved':<10} {'Status':<30}")
    print(f"{'-'*90}")
    
    total_records = 0
    total_failed = 0
    total_preserved = 0
    
    for stats in stats_list:
        dataset = stats.get("dataset", stats.get("file", "unknown"))
        total = stats["total"]
        failed = stats["failed"]
        preserved = stats["preserved"]
        status = stats["status"]
        
        print(f"{dataset:<20} {total:<8} {failed:<8} {preserved:<10} {status:<30}")
        
        if "skipped" not in status and "already processed" not in status:
            total_records += total
            total_failed += failed
            total_preserved += preserved
    
    print(f"{'-'*90}")
    print(f"{'TOTAL':<20} {total_records:<8} {total_failed:<8} {total_preserved:<10}")
    print(f"\n{'='*80}")
    
    if args.dry_run:
        print("DRY-RUN complete. Run without --dry-run to apply changes.")
    else:
        converted_count = sum(1 for s in stats_list if s["status"] == "converted")
        print(f"Converted {converted_count} datasets to checkpoint format.")
        print(f"Original files renamed to *-pre-repair.json")
        print(f"Checkpoint files created: *-checkpoint.json")
    
    return 0


if __name__ == "__main__":
    exit(main())
