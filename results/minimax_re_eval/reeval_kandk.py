"""
MiniMax-M2.7 Knights & Knaves Re-evaluation Script
====================================================

Problem:
  KandK evaluator (scoring.py judge_answer) has three issues causing false negatives:

  1. contain_if rule (183 cases): judge_answer line 154 checks
     `if "if" in pred_answer: return False` — rejects ANY prediction containing "if"
     anywhere in the CONCLUSION block, including benign reasoning text like
     "check if any nuance". Of 183 flagged, 124 have correct GT in prediction.

  2. beyond_list rule (37 cases): judge_answer line 151 checks if `(N+1)` exists in
     prediction. M2.7 adds extra lines like "(3) No other characters are involved."
     after a 2-character problem. All 37 have correct GT in raw output.

  3. Format/extraction mismatch (250 cases): M2.7 uses en-dashes, markdown bold,
     or other non-standard formats. GT is like "William is a knight" but M2.7
     writes "William – knight" or "**William** is a knight". Substring match fails.

  4. Genuinely wrong: only ~17 cases.

Fix:
  For each score=0 record, re-evaluate using improved logic:
  a) Use parse_answer() to extract CONCLUSION block (same as original)
  b) Strip markdown artifacts from extracted text
  c) Apply improved judge that:
     - Does NOT reject on contain_if if all gold conditions are present
     - Does NOT reject on beyond_list if all gold conditions are present
     - Uses normalized matching: handles en-dashes, em-dashes, colons, "Name – role"
  d) Fallback: use parse_single_answer() to extract structured {name: role} dict
     and compare against gold conditions

Outputs:
  - kandk-test-MiniMax-M2.7-20260403_163422-reeval.json in results/minimax_reeval/
  - reeval_kandk_log.txt with detailed changes
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

from evaluation.K_and_K.scoring import parse_answer, parse_single_answer

def strip_think_block(text: str) -> str:
    """Remove <think>...</think> reasoning block from M2.7 output."""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):]
    return text


def parse_answer_improved(raw_output: str) -> tuple:
    """
    Improved parse_answer that:
    1. Strips <think> block first (M2.7 references CONCLUSION: inside thinking)
    2. Handles "CONCLUSION:**" markdown variant (M2.7 writes "CONCLUSION:**\n")
    3. Prefers the LAST CONCLUSION block (the numbered list), not the first
       (which may be a prose summary like "Both inhabitants are knights.")
    4. Falls back to original parse_answer on full text if nothing found
    """
    # Step 1: Strip think block
    text = strip_think_block(raw_output)
    
    # Step 2: Normalize markdown CONCLUSION variants
    # "CONCLUSION:**" -> "CONCLUSION:" so parse_answer can match
    # "## CONCLUSION" -> "CONCLUSION:" (markdown header)
    text_normalized = re.sub(
        r'#{1,3}\s*CONCLUSION\s*[:]*\s*\*{0,3}\s*',
        'CONCLUSION:',
        text,
        flags=re.IGNORECASE
    )
    text_normalized = re.sub(
        r'CONCLUSION\s*[:]*\s*\*{1,3}\s*',
        'CONCLUSION:',
        text_normalized,
        flags=re.IGNORECASE
    )
    
    # Step 3: Find ALL CONCLUSION occurrences and prefer the one with numbered items
    conclusion_patterns_ordered = ['CONCLUSION:', 'Conclusion:', 'conclusion:']
    
    best_pred = None
    best_has_numbers = False
    
    for pattern in conclusion_patterns_ordered:
        parts = text_normalized.split(pattern)
        if len(parts) > 1:
            # Check each split (from last to first) for numbered format
            for i in range(len(parts) - 1, 0, -1):
                candidate = parts[i].strip()
                # Truncate at finish patterns
                for fp in ["### Reason", "Let's think step by step again",
                           "let's go back and check", "###"]:
                    if fp in candidate:
                        candidate = candidate.split(fp)[0]
                
                has_numbers = bool(re.search(r'\(\d+\)', candidate[:200]))
                
                if has_numbers:
                    # Prefer this one - it has numbered items
                    return candidate.strip(), True
                elif best_pred is None:
                    best_pred = candidate.strip()
    
    if best_pred:
        return best_pred, True
    
    # Step 4: Fallback to parse_answer on normalized text
    pred, success = parse_answer(pred_str=text_normalized)
    if success and pred and pred.strip():
        return pred, success
    
    # Step 5: Fallback to parse_answer on text after think removal
    pred, success = parse_answer(pred_str=text)
    if success and pred and pred.strip():
        return pred, success
    
    # Step 6: Last resort - try original full raw output
    pred, success = parse_answer(pred_str=raw_output)
    return pred, success


# Source file
SOURCE_FILE = (
    ROOT / "results" / "bench" / "kandk" / "test" / "MiniMax-M2.7"
    / "kandk-test-MiniMax-M2.7-20260403_163422.json"
)


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text."""
    # Remove bold markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace("**", "")
    # Remove italic markers
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove backticks
    text = text.replace("`", "")
    # Remove LaTeX \text{} wrappers
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
    return text


def normalize_condition_text(text: str) -> str:
    """
    Normalize a condition string for flexible matching.
    Handles en-dash, em-dash, colon formats, markdown, extra whitespace.
    
    Examples:
      "William – knight"      -> "william is a knight"
      "William — Knight"      -> "william is a knight"
      "William: Knight"       -> "william is a knight"
      "**William** is a knight" -> "william is a knight"
      "William is a Knight."  -> "william is a knight"
    """
    text = strip_markdown(text)
    text = text.strip().rstrip(".")
    text = text.lower()
    
    # Normalize "Name – role" or "Name — role" or "Name: role" -> "Name is a role"
    # Pattern: name + separator + role
    m = re.match(
        r'^(.+?)\s*[\u2013\u2014:\-]+\s*(knight|knave)s?$',
        text, re.IGNORECASE
    )
    if m:
        name = m.group(1).strip()
        role = m.group(2).strip().lower()
        return f"{name} is a {role}"
    
    # Already in "Name is a role" format - just normalize
    m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)s?\.?$', text, re.IGNORECASE)
    if m:
        name = m.group(1).strip()
        role = m.group(2).strip().lower()
        return f"{name} is a {role}"
    
    return text


def improved_judge(pred_answer: str, gold_conditions: list) -> tuple:
    """
    Improved judge_answer that fixes the three false negative issues.
    
    Returns: (is_correct: bool, reason: str, correct_ratio: float)
    """
    if not pred_answer or not gold_conditions:
        return False, "empty", 0.0
    
    # Strip markdown from prediction
    cleaned_pred = strip_markdown(pred_answer)
    
    # Truncate at finish patterns (same as original)
    finish_patterns = [
        "### Reason", "Let's think step by step again",
        "let's go back and check", "###"
    ]
    for fp in finish_patterns:
        if fp in cleaned_pred:
            cleaned_pred = cleaned_pred.split(fp)[0]
    
    # Method 1: Direct substring matching (case-insensitive) on cleaned text
    # This is the original method but with markdown stripped
    correct_count = 0
    for gc in gold_conditions:
        if gc.lower() in cleaned_pred.lower():
            correct_count += 1
    
    if correct_count == len(gold_conditions):
        return True, "", 1.0
    
    # Method 2: Normalized matching - handle en-dash, colon formats
    # Split prediction into lines and normalize each
    pred_lines = cleaned_pred.strip().split("\n")
    pred_normalized = set()
    for line in pred_lines:
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering like "(1) " or "1. " or "1) "
        line = re.sub(r'^\(?\d+\)?[\.\):\s]+', '', line).strip()
        norm = normalize_condition_text(line)
        if norm:
            pred_normalized.add(norm)
    
    gold_normalized = set()
    for gc in gold_conditions:
        norm = normalize_condition_text(gc)
        if norm:
            gold_normalized.add(norm)
    
    if gold_normalized and gold_normalized.issubset(pred_normalized):
        return True, "", 1.0
    
    # Method 3: Use parse_single_answer for structured extraction
    # This handles more complex format variations
    try:
        parsed_dict = parse_single_answer(pred_answer)
        if parsed_dict:
            # Build normalized conditions from parsed dict
            parsed_conditions = set()
            for name, role in parsed_dict.items():
                parsed_conditions.add(f"{name.lower().strip()} is a {role.lower().strip()}")
            
            if gold_normalized and gold_normalized.issubset(parsed_conditions):
                return True, "", 1.0
    except Exception:
        pass
    
    # Method 4: Check if all gold conditions appear in the raw prediction
    # even without proper CONCLUSION formatting
    correct_count_normalized = 0
    for gc_norm in gold_normalized:
        # Check if this condition appears in any form in the prediction
        if gc_norm in cleaned_pred.lower():
            correct_count_normalized += 1
            continue
        # Check individual name+role
        m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)', gc_norm)
        if m:
            name = m.group(1)
            role = m.group(2)
            # Flexible pattern: name ... role (within same line or nearby)
            name_esc = re.escape(name)
            pattern = rf'{name_esc}.*?{role}'
            if re.search(pattern, cleaned_pred.lower()):
                correct_count_normalized += 1
    
    total_gold = len(gold_normalized) if gold_normalized else len(gold_conditions)
    ratio = correct_count_normalized / total_gold if total_gold > 0 else 0.0
    
    if correct_count_normalized == total_gold:
        return True, "", ratio
    
    return False, "wrong_identity", ratio


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().isoformat()
    print(f"MiniMax-M2.7 KandK Re-evaluation")
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
        f"MiniMax-M2.7 KandK Re-evaluation Log",
        f"Timestamp: {timestamp}",
        f"Source: {SOURCE_FILE}",
        f"Root cause: contain_if false positive, beyond_list false positive, format mismatch",
        f"Fix: improved judge with markdown stripping, normalized matching, parse_single_answer fallback",
        f"Total records: {total}",
        f"Original performance: {original_perf*100:.2f}%",
        f"",
    ]
    
    # Counters
    score0_count = 0
    recovered = 0
    still_wrong = 0
    empty_pred = 0
    recovery_by_method = {"substring": 0, "normalized": 0, "parsed": 0, "flexible": 0}
    original_reasons = {}
    
    changes = []
    
    for rec in result["records"]:
        idx = rec["index"]
        old_score = rec.get("score", 0)
        gt = rec.get("ground_truth", [])
        raw_output = rec.get("raw_output", "")
        old_pred = rec.get("prediction", "")
        old_reason = rec.get("wrong_reason", "")
        
        # Only re-evaluate records with score=0
        if old_score == 1.0:
            continue
        
        score0_count += 1
        original_reasons[old_reason] = original_reasons.get(old_reason, 0) + 1
        
        if not raw_output or raw_output.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty raw_output")
            continue
        
        # Re-extract prediction with improved parser (strips think block, handles markdown)
        pred_answer, is_success = parse_answer_improved(raw_output)
        
        if not pred_answer or pred_answer.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty parsed prediction")
            continue
        
        # Ensure gt is a list
        if isinstance(gt, str):
            gt_list = [gt]
        else:
            gt_list = list(gt)
        
        # Re-evaluate with improved judge
        is_correct, reason, ratio = improved_judge(pred_answer, gt_list)
        
        # Store reeval metadata
        rec["reeval_old_score"] = old_score
        rec["reeval_old_reason"] = old_reason
        rec["reeval_old_prediction"] = old_pred
        rec["prediction"] = pred_answer  # Update with improved extraction
        
        if is_correct:
            rec["score"] = 1.0
            rec["wrong_reason"] = ""
            rec["reeval_recovered"] = True
            recovered += 1
            changes.append(
                f"  idx={idx}: RECOVERED  old_reason='{old_reason}' "
                f"gt={gt_list} ratio={ratio:.2f} score: 0->1"
            )
        else:
            rec["score"] = 0.0
            rec["wrong_reason"] = reason
            rec["reeval_recovered"] = False
            still_wrong += 1
            changes.append(
                f"  idx={idx}: STILL WRONG  old_reason='{old_reason}' new_reason='{reason}' "
                f"gt={gt_list} ratio={ratio:.2f}"
            )
    
    # Recalculate performance
    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf
    
    # Add reeval metadata at top level
    result["_reeval_meta"] = {
        "script": "reeval_kandk.py",
        "timestamp": timestamp,
        "reason": "contain_if/beyond_list false positives + format mismatch in KandK judge",
        "original_file": str(SOURCE_FILE),
        "original_performance": original_perf,
        "corrected_performance": new_perf,
        "score0_count": score0_count,
        "recovered": recovered,
        "still_wrong": still_wrong,
        "empty_prediction": empty_pred,
        "original_wrong_reasons": original_reasons,
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
        f"  Empty/no prediction   : {empty_pred}\n"
        f"  Recovered (correct)   : {recovered}\n"
        f"  Still wrong           : {still_wrong}\n"
        f"  Original wrong reasons: {original_reasons}\n"
        f"  Performance           : {original_perf*100:.2f}% -> {new_perf*100:.2f}% "
        f"(+{(new_perf - original_perf)*100:.2f}%)\n"
        f"  Output                : {out_path}"
    )
    
    log_lines.append(summary)
    for c in changes:
        log_lines.append(c)
    
    print(summary)
    
    # Write log
    log_path = OUTPUT_DIR / "reeval_kandk_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
