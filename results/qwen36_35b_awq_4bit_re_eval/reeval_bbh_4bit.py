"""
qwen3.6-35b-awq-4bit BBH Re-evaluation Script
=============================================

Ported from MiniMax-M2.7 BBH re-eval logic for qwen3.6-35b-awq-4bit.
"""

import copy
import json
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "qwen36_35b_awq_4bit_re_eval"

sys.path.insert(0, str(ROOT))
from evaluation.deepscaler_rm import grade_answer_mathd, grade_answer_sympy

SOURCE_FILE = (
    ROOT / "results" / "bench" / "bbh" / "test" / "qwen3.6-35b-awq-4bit"
    / "bbh-test-qwen3.6-35b-awq-4bit-20260420_182324.json"
)


def clean_prediction(pred: str) -> str:
    """Strip markdown artifacts from BBH predictions."""
    if not pred:
        return pred

    cleaned = pred.strip()
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("*", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r'^(?:Answer|ANSWER|answer)\s*:\s*', '', cleaned)
    cleaned = cleaned.strip().rstrip(".")
    return cleaned


def grade_answer(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False

    if grade_answer_mathd(prediction, ground_truth):
        return True
    if grade_answer_sympy(prediction, ground_truth):
        return True

    pred_norm = prediction.strip().lower()
    gt_norm = ground_truth.strip().lower()
    if pred_norm == gt_norm:
        return True

    pred_no_parens = pred_norm.strip("()")
    gt_no_parens = gt_norm.strip("()")
    if pred_no_parens == gt_no_parens and len(pred_no_parens) > 0:
        return True

    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    print("qwen3.6-35b-awq-4bit BBH Re-evaluation")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Timestamp : {timestamp}")
    print(f"Source    : {SOURCE_FILE}\n")

    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        sys.exit(1)

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        original = json.load(f)

    result = copy.deepcopy(original)
    total = len(result["records"])
    original_perf = original["performance"]

    log_lines = [
        "qwen3.6-35b-awq-4bit BBH Re-evaluation Log",
        f"Timestamp: {timestamp}",
        f"Source: {SOURCE_FILE}",
        "Root cause: markdown artifacts in extracted BBH predictions",
        "Fix: strip markdown artifacts then re-grade with mathd+sympy+exact match",
        f"Total records: {total}",
        f"Original performance: {original_perf * 100:.2f}%",
        "",
    ]

    score0_count = 0
    recovered = 0
    still_wrong = 0
    already_clean = 0
    empty_pred = 0
    changes = []

    for rec in result["records"]:
        idx = rec["index"]
        old_pred = rec.get("prediction", "")
        old_score = rec.get("score", 0)
        gt = rec.get("ground_truth", "")

        if old_score == 1.0:
            continue

        score0_count += 1

        if not old_pred or old_pred.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty prediction")
            continue

        cleaned_pred = clean_prediction(old_pred)

        if cleaned_pred == old_pred.strip():
            already_clean += 1
            changes.append(f"  idx={idx}: ALREADY CLEAN pred='{old_pred}' gt='{gt}' -> genuinely wrong")
            continue

        is_correct = grade_answer(cleaned_pred, gt)

        rec["reeval_old_prediction"] = old_pred
        rec["reeval_old_score"] = old_score
        rec["reeval_cleaned_prediction"] = cleaned_pred

        if is_correct:
            rec["prediction"] = cleaned_pred
            rec["score"] = 1.0
            recovered += 1
            changes.append(f"  idx={idx}: RECOVERED '{old_pred}' -> '{cleaned_pred}' gt='{gt}' score: 0->1")
        else:
            rec["prediction"] = cleaned_pred
            rec["score"] = 0.0
            still_wrong += 1
            changes.append(f"  idx={idx}: CLEANED but wrong '{old_pred}' -> '{cleaned_pred}' gt='{gt}' score: 0->0")

    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf

    result["_reeval_meta"] = {
        "script": "reeval_bbh_4bit.py",
        "timestamp": timestamp,
        "reason": "markdown artifacts (**, `, Answer: prefix) in extracted BBH predictions",
        "original_file": str(SOURCE_FILE),
        "original_performance": original_perf,
        "corrected_performance": new_perf,
        "score0_count": score0_count,
        "recovered": recovered,
        "still_wrong_after_cleaning": still_wrong,
        "already_clean_genuinely_wrong": already_clean,
        "empty_prediction": empty_pred,
    }

    out_path = OUTPUT_DIR / f"{SOURCE_FILE.stem}-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = (
        f"\nSummary:\n"
        f"  Total records         : {total}\n"
        f"  Score=0 records       : {score0_count}\n"
        f"  Empty prediction      : {empty_pred}\n"
        f"  Already clean (wrong) : {already_clean}\n"
        f"  Recovered (correct)   : {recovered}\n"
        f"  Cleaned but wrong     : {still_wrong}\n"
        f"  Performance           : {original_perf * 100:.2f}% -> {new_perf * 100:.2f}% "
        f"(+{(new_perf - original_perf) * 100:.2f}%)\n"
        f"  Output                : {out_path}"
    )

    log_lines.append(summary)
    log_lines.extend(changes)

    print(summary)

    log_path = OUTPUT_DIR / "reeval_bbh_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
