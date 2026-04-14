"""
MiniMax-M2.7 BBH Re-evaluation Script
======================================

Problem:
  BBH evaluator uses ANSWER_PATTERN = r"(?i)Answer\s*:\s*([^\n]+)" to extract
  the answer from raw output. The regex itself works (captures everything after
  "Answer:"), but M2.7 wraps answers in markdown bold: **Answer: (B)**
  
  The extraction produces predictions like "** False", "(B)`", "True**" etc.
  These polluted strings are then fed to grade_answer_mathd / grade_answer_sympy
  (math graders from deepscaler_rm.py) which call mathd_normalize_answer().
  The markdown characters cause normalization mismatches.

  Analysis of 578 score=0 records:
    - 504 recoverable by stripping markdown artifacts (**, `, "Answer:" prefix)
    - 74 genuinely wrong

Fix:
  For each score=0 record, take the stored `prediction` field, strip markdown
  artifacts, then re-grade using the same grade_answer_mathd / grade_answer_sympy
  functions (for consistency with original evaluator), PLUS a fallback
  case-insensitive exact match for simple text answers (True/False, Yes/No, etc.)

Outputs:
  - bbh-test-MiniMax-M2.7-20260410_043359-reeval.json in results/minimax_reeval/
  - reeval_bbh_log.txt with detailed changes
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

# Add project root to path so we can import evaluation modules
sys.path.insert(0, str(ROOT))

from evaluation.deepscaler_rm import grade_answer_mathd, grade_answer_sympy

# Source file
SOURCE_FILE = (
    ROOT / "results" / "bench" / "bbh" / "test" / "MiniMax-M2.7"
    / "bbh-test-MiniMax-M2.7-20260410_043359.json"
)


def clean_prediction(pred: str) -> str:
    """
    Strip markdown artifacts from BBH predictions.
    
    Known pollution patterns from M2.7:
      - "** False"   -> "False"
      - "True**"     -> "True"
      - "(B)`"       -> "(B)"
      - "**(B)**"    -> "(B)"
      - "**Answer:** (B)" -> "(B)"  (double extraction artifact)
    """
    if not pred:
        return pred
    
    cleaned = pred.strip()
    
    # Remove markdown bold/italic markers
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("*", "")
    
    # Remove backticks
    cleaned = cleaned.replace("`", "")
    
    # Remove "Answer:" prefix that sometimes gets included
    cleaned = re.sub(r'^(?:Answer|ANSWER|answer)\s*:\s*', '', cleaned)
    
    # Remove leading/trailing whitespace and periods
    cleaned = cleaned.strip().rstrip(".")
    
    return cleaned


def grade_answer(prediction: str, ground_truth: str) -> bool:
    """
    Grade a BBH answer using the same logic as the original evaluator,
    plus a fallback for simple text answers.
    
    Original evaluator uses: grade_answer_mathd OR grade_answer_sympy
    We add: case-insensitive exact match for text answers (True/False, Yes/No, etc.)
    """
    if not prediction or not ground_truth:
        return False
    
    # 1. Try original grading (mathd normalization + sympy)
    if grade_answer_mathd(prediction, ground_truth):
        return True
    if grade_answer_sympy(prediction, ground_truth):
        return True
    
    # 2. Fallback: case-insensitive exact match (handles True/False, Yes/No,
    #    Valid/Invalid, option letters like "(A)", etc.)
    pred_norm = prediction.strip().lower()
    gt_norm = ground_truth.strip().lower()
    if pred_norm == gt_norm:
        return True
    
    # 3. Handle parenthesized options: "(B)" == "(B)" already handled above,
    #    but also try stripping parens: "B" == "(B)" or "(B)" == "B"
    pred_no_parens = pred_norm.strip("()")
    gt_no_parens = gt_norm.strip("()")
    if pred_no_parens == gt_no_parens and len(pred_no_parens) > 0:
        return True
    
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    print(f"MiniMax-M2.7 BBH Re-evaluation")
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
        f"MiniMax-M2.7 BBH Re-evaluation Log",
        f"Timestamp: {timestamp}",
        f"Source: {SOURCE_FILE}",
        f"Root cause: markdown artifacts (**, `, Answer: prefix) in extracted predictions",
        f"Fix: strip markdown artifacts then re-grade with mathd+sympy+exact match",
        f"Total records: {total}",
        f"Original performance: {original_perf*100:.2f}%",
        f"",
    ]
    
    # Counters
    score0_count = 0
    recovered = 0
    still_wrong = 0
    already_clean = 0  # prediction was already clean, genuinely wrong
    empty_pred = 0
    
    changes = []
    
    for rec in result["records"]:
        idx = rec["index"]
        old_pred = rec.get("prediction", "")
        old_score = rec.get("score", 0)
        gt = rec.get("ground_truth", "")
        
        # Only re-evaluate records with score=0
        if old_score == 1.0:
            continue
        
        score0_count += 1
        
        if not old_pred or old_pred.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty prediction")
            continue
        
        # Clean the prediction
        cleaned_pred = clean_prediction(old_pred)
        
        if cleaned_pred == old_pred.strip():
            # No change from cleaning - genuinely wrong answer
            already_clean += 1
            changes.append(f"  idx={idx}: ALREADY CLEAN pred='{old_pred}' gt='{gt}' -> genuinely wrong")
            continue
        
        # Re-grade with cleaned prediction
        is_correct = grade_answer(cleaned_pred, gt)
        
        # Store reeval metadata
        rec["reeval_old_prediction"] = old_pred
        rec["reeval_old_score"] = old_score
        rec["reeval_cleaned_prediction"] = cleaned_pred
        
        if is_correct:
            rec["prediction"] = cleaned_pred
            rec["score"] = 1.0
            recovered += 1
            changes.append(f"  idx={idx}: RECOVERED  '{old_pred}' -> '{cleaned_pred}' gt='{gt}' score: 0->1")
        else:
            rec["prediction"] = cleaned_pred
            rec["score"] = 0.0
            still_wrong += 1
            changes.append(f"  idx={idx}: CLEANED but wrong  '{old_pred}' -> '{cleaned_pred}' gt='{gt}' score: 0->0")
    
    # Recalculate performance
    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf
    
    # Add reeval metadata at top level
    result["_reeval_meta"] = {
        "script": "reeval_bbh.py",
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
    
    # Write output
    out_name = f"{SOURCE_FILE.stem}-reeval.json"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Summary
    summary = (
        f"\nSummary:\n"
        f"  Total records         : {total}\n"
        f"  Score=0 records       : {score0_count}\n"
        f"  Empty prediction      : {empty_pred}\n"
        f"  Already clean (wrong) : {already_clean}\n"
        f"  Recovered (correct)   : {recovered}\n"
        f"  Cleaned but wrong     : {still_wrong}\n"
        f"  Performance           : {original_perf*100:.2f}% -> {new_perf*100:.2f}% "
        f"(+{(new_perf - original_perf)*100:.2f}%)\n"
        f"  Output                : {out_path}"
    )
    
    log_lines.append(summary)
    for c in changes:
        log_lines.append(c)
    
    print(summary)
    
    # Write log
    log_path = OUTPUT_DIR / "reeval_bbh_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
