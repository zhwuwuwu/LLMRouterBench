"""
MiniMax-M2.7 MCQ Re-evaluation Script
======================================

Problem:
  The evaluator's ANSWER_PATTERN regex fails on M2.7's markdown bold output
  format: **Answer:** D
  
  Regex: r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?"
  On "**Answer:** D", after matching "Answer:", the regex encounters "**"
  (markdown bold closing) instead of whitespace/letter -> NO MATCH.
  
  This causes 145 MCQ records across 7 datasets to get empty prediction
  and score=0 despite having valid answers in raw_output.

Fix:
  Strip markdown bold/italic markers around the Answer keyword and between
  the colon and answer letter before applying the original regex.

Affected datasets (145 records):
  meld(58), emorynlp(34), arcc(23), medqa(17), mmlupro(11),
  winogrande(1), gpqa(1)

Note: livecodebench has 1 empty-prediction record but it's a coding task,
  not MCQ, so it's handled by reeval_minimax_coding.py instead.

Outputs:
  - Per-dataset reeval JSON files in results/minimax_reeval/
  - Consolidated reeval_mcq_log.txt with all changes
"""

import copy
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]  # LLMRouterBench/
OUTPUT_DIR = ROOT / "results" / "minimax_reeval"

# ---------------------------------------------------------------------------
# Dataset configurations
# Each entry: (dataset_name, result_file_path, answer_pattern, split)
# answer_pattern is the ORIGINAL pattern from the evaluator source
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "meld",
        "path": ROOT / "results" / "bench" / "meld" / "test" / "MiniMax-M2.7"
                / "meld-test-MiniMax-M2.7-20260410_093630.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?",
    },
    {
        "name": "emorynlp",
        "path": ROOT / "results" / "bench" / "emorynlp" / "test" / "MiniMax-M2.7"
                / "emorynlp-test-MiniMax-M2.7-20260410_080908.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?",
    },
    {
        "name": "arcc",
        "path": ROOT / "results" / "bench" / "arcc" / "test" / "MiniMax-M2.7"
                / "arcc-test-MiniMax-M2.7-20260410_070718.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-E])[.\s\n]?",
    },
    {
        "name": "medqa",
        "path": ROOT / "results" / "bench" / "medqa" / "test" / "MiniMax-M2.7"
                / "medqa-test-MiniMax-M2.7-20260410_065433.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?",
    },
    {
        "name": "mmlupro",
        "path": ROOT / "results" / "bench" / "mmlupro" / "test_3000" / "MiniMax-M2.7"
                / "mmlupro-test_3000-MiniMax-M2.7-20260410_041143.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-J])[.\s\n]?",
    },
    {
        "name": "winogrande",
        "path": ROOT / "results" / "bench" / "winogrande" / "valid" / "MiniMax-M2.7"
                / "winogrande-valid-MiniMax-M2.7-20260410_050752.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-B])[.\s\n]?",
    },
    {
        "name": "gpqa",
        "path": ROOT / "results" / "bench" / "gpqa" / "test" / "MiniMax-M2.7"
                / "gpqa-test-MiniMax-M2.7-20260410_014220.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?",
    },
]


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning block."""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):]
    return text


def extract_answer_fixed(text: str, answer_pattern: str) -> str:
    """
    Fixed answer extraction that handles M2.7's markdown bold format.

    Handles these known formats:
      - **Answer:** D        (142 records - markdown bold keyword)
      - **Answer:** **B**.   (1 record  - double bold)
      - Answer: E            (1 record  - letter out of pattern range, handled by caller)
      - (no answer line)     (2 records - truly empty, returns "")

    Strategy: strip all markdown bold/italic markers (* sequences) from the
    text around Answer/answer lines, then apply the original regex.
    """
    if not text:
        return ""

    # Strip <think> block first
    cleaned = strip_think(text)

    # Strip markdown bold/italic markers globally around answer-related content
    # 1. Remove ** or *** wrapping the word "Answer"
    cleaned = re.sub(r'\*{1,3}\s*(Answer)\s*\*{1,3}', r'\1', cleaned, flags=re.IGNORECASE)
    # 2. Remove ** between colon and the answer letter (e.g., ": **B**." -> ": B.")
    cleaned = re.sub(r':\s*\*{1,3}\s*', ': ', cleaned)
    # 3. Remove trailing ** after the answer letter (e.g., "B**." -> "B.")
    cleaned = re.sub(r'(\b[A-J])\*{1,3}', r'\1', cleaned)

    # Apply original regex
    matches = re.findall(answer_pattern, cleaned)
    if matches:
        return matches[-1].strip()

    # Fallback: try "Final Answer" variant (same as base_evaluator.py)
    final_pattern = answer_pattern.replace(r"Answer\s*:\s", r"Final Answer\s\n+\s")
    final_match = re.search(final_pattern, cleaned)
    if final_match:
        return final_match.group(1).strip()

    return ""


def reeval_dataset(config: dict, log_lines: list[str]) -> dict:
    """
    Re-evaluate a single MCQ dataset.
    Returns summary dict with counts and performance delta.
    """
    name = config["name"]
    result_path = config["path"]
    answer_pattern = config["answer_pattern"]

    log_lines.append(f"\n{'='*60}")
    log_lines.append(f"Dataset: {name}")
    log_lines.append(f"Source : {result_path}")
    log_lines.append(f"Pattern: {answer_pattern}")
    log_lines.append(f"{'='*60}")

    with open(result_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    result = copy.deepcopy(original)
    total = len(result["records"])
    original_perf = original["performance"]

    # Counters
    empty_pred_count = 0
    recovered = 0
    still_empty = 0
    wrong_answer = 0
    skipped_gen_fail = 0

    changes = []  # detailed per-record log

    for rec in result["records"]:
        idx = rec["index"]
        old_pred = rec.get("prediction", "")
        old_score = rec.get("score", 0)
        raw = rec.get("raw_output", "")
        gt = rec.get("ground_truth", "")

        # Only re-evaluate records with empty prediction and non-empty raw_output
        if old_pred != "" or old_score == 1.0:
            continue

        if not raw or raw.strip() == "":
            # Generation failed - no raw output to re-extract from
            skipped_gen_fail += 1
            continue

        empty_pred_count += 1

        # Try fixed extraction
        new_pred = extract_answer_fixed(raw, answer_pattern)

        if not new_pred:
            still_empty += 1
            changes.append(f"  idx={idx}: still empty after fix (no answer line in output)")
            continue

        # Store reeval metadata on the record
        rec["reeval_old_prediction"] = old_pred
        rec["reeval_old_score"] = old_score

        if new_pred == gt:
            rec["prediction"] = new_pred
            rec["score"] = 1.0
            recovered += 1
            changes.append(f"  idx={idx}: RECOVERED  pred='{new_pred}' gt='{gt}' score: 0->1")
        else:
            rec["prediction"] = new_pred
            rec["score"] = 0.0  # still wrong, but now has a prediction
            wrong_answer += 1
            changes.append(f"  idx={idx}: EXTRACTED but wrong  pred='{new_pred}' gt='{gt}' score: 0->0")

    # Recalculate performance
    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf

    # Add reeval metadata at top level
    result["_reeval_meta"] = {
        "script": "reeval_mcq.py",
        "timestamp": datetime.now().isoformat(),
        "reason": "ANSWER_PATTERN regex fails on **Answer:** X markdown bold format",
        "original_file": str(result_path),
        "original_performance": original_perf,
        "corrected_performance": new_perf,
        "empty_pred_found": empty_pred_count,
        "recovered": recovered,
        "still_empty": still_empty,
        "wrong_answer_extracted": wrong_answer,
        "skipped_gen_fail": skipped_gen_fail,
    }

    # Write output
    out_name = f"{name}-{result_path.stem.split(name + '-')[1]}-reeval.json" if name in result_path.stem else f"{name}-reeval.json"
    # Simpler: use dataset name + model
    out_name = f"{result_path.stem}-reeval.json"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Log summary
    summary = (
        f"\n  Summary for {name}:\n"
        f"    Total records      : {total}\n"
        f"    Empty predictions  : {empty_pred_count}\n"
        f"    Recovered (correct): {recovered}\n"
        f"    Extracted (wrong)  : {wrong_answer}\n"
        f"    Still empty        : {still_empty}\n"
        f"    Skipped (gen fail) : {skipped_gen_fail}\n"
        f"    Performance        : {original_perf*100:.2f}% -> {new_perf*100:.2f}% "
        f"(+{(new_perf - original_perf)*100:.2f}%)\n"
        f"    Output             : {out_path}"
    )
    log_lines.append(summary)
    for c in changes:
        log_lines.append(c)

    print(f"  {name:12s}: {empty_pred_count} empty -> {recovered} recovered, "
          f"{wrong_answer} wrong, {still_empty} still empty | "
          f"{original_perf*100:.2f}% -> {new_perf*100:.2f}%")

    return {
        "name": name,
        "total": total,
        "empty_pred": empty_pred_count,
        "recovered": recovered,
        "wrong_answer": wrong_answer,
        "still_empty": still_empty,
        "original_perf": original_perf,
        "new_perf": new_perf,
        "output_file": str(out_path),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MiniMax-M2.7 MCQ Re-evaluation")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Datasets  : {len(DATASETS)}\n")

    log_lines = [
        f"MiniMax-M2.7 MCQ Re-evaluation Log",
        f"Timestamp: {datetime.now().isoformat()}",
        f"Root cause: ANSWER_PATTERN regex fails on **Answer:** X markdown bold format",
        f"Fix: strip markdown markers before regex extraction",
    ]

    all_summaries = []
    total_recovered = 0
    total_empty = 0
    total_wrong = 0
    total_still_empty = 0

    for config in DATASETS:
        if not config["path"].exists():
            msg = f"WARNING: {config['name']} result file not found: {config['path']}"
            print(f"  {msg}")
            log_lines.append(f"\n{msg}")
            continue

        summary = reeval_dataset(config, log_lines)
        all_summaries.append(summary)
        total_recovered += summary["recovered"]
        total_empty += summary["empty_pred"]
        total_wrong += summary["wrong_answer"]
        total_still_empty += summary["still_empty"]

    # Grand summary
    grand = (
        f"\n\n{'='*60}\n"
        f"GRAND SUMMARY\n"
        f"{'='*60}\n"
        f"  Total empty predictions found : {total_empty}\n"
        f"  Total recovered (correct)     : {total_recovered}\n"
        f"  Total extracted (wrong answer) : {total_wrong}\n"
        f"  Total still empty             : {total_still_empty}\n"
        f"\n  Per-dataset performance changes:"
    )
    log_lines.append(grand)
    print(f"\n{'='*60}")
    print(f"GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"  Total empty: {total_empty} -> Recovered: {total_recovered}, "
          f"Wrong: {total_wrong}, Still empty: {total_still_empty}")

    for s in all_summaries:
        line = (f"    {s['name']:12s}: {s['original_perf']*100:.2f}% -> "
                f"{s['new_perf']*100:.2f}% (+{(s['new_perf']-s['original_perf'])*100:.2f}%)")
        log_lines.append(line)
        print(line)

    # Write log file
    log_path = OUTPUT_DIR / "reeval_mcq_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
