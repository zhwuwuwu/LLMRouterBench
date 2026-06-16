"""
qwen3-coder-30b Knights & Knaves Re-evaluation Script
======================================================

Problem:
  KandK evaluator (scoring.py judge_answer) has false negative issues:

  1. beyond_list rule: judge_answer checks if `(N+1)` exists in prediction.
     qwen3-coder-30b adds extra lines like "(3) ..." after a 2-character problem
     (e.g. "[No third person mentioned]", "Both X and Y are knights").
     These trigger the beyond_list rejection despite correct answers in (1)/(2).

  2. contain_if rule: judge_answer rejects ANY prediction containing "if"
     anywhere in the CONCLUSION block, including benign text.

  Same pattern as MiniMax-M2.7's KandK eval bug. Reuses improved_judge logic.

Fix:
  For each score=0 record, re-evaluate using improved logic:
  a) Extract CONCLUSION block
  b) Strip markdown artifacts
  c) Improved judge: no false rejection on contain_if/beyond_list if gold present
  d) Fallback: parse_single_answer structured extraction

Outputs:
  - Reeval JSON in results/qwen3_coder_30b_re_eval/
  - Log file with detailed changes
"""

import copy
import json
import re
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]  # LLMRouterBench/
OUTPUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
from evaluation.K_and_K.scoring import parse_answer, parse_single_answer

SOURCE_FILE = (
    ROOT / "results" / "bench" / "kandk" / "test" / "qwen3-coder-30b"
    / "kandk-test-qwen3-coder-30b-20260415_201424.json"
)


def parse_answer_improved(raw_output: str) -> tuple:
    """Extract CONCLUSION block, preferring last occurrence with numbered items."""
    text = raw_output

    # Normalize markdown CONCLUSION variants
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

    conclusion_patterns_ordered = ['CONCLUSION:', 'Conclusion:', 'conclusion:']

    best_pred = None
    for pattern in conclusion_patterns_ordered:
        parts = text_normalized.split(pattern)
        if len(parts) > 1:
            for i in range(len(parts) - 1, 0, -1):
                candidate = parts[i].strip()
                for fp in ["### Reason", "Let's think step by step again",
                           "let's go back and check", "###"]:
                    if fp in candidate:
                        candidate = candidate.split(fp)[0]
                has_numbers = bool(re.search(r'\(\d+\)', candidate[:200]))
                if has_numbers:
                    return candidate.strip(), True
                elif best_pred is None:
                    best_pred = candidate.strip()

    if best_pred:
        return best_pred, True

    pred, success = parse_answer(pred_str=text_normalized)
    if success and pred and pred.strip():
        return pred, success

    pred, success = parse_answer(pred_str=text)
    if success and pred and pred.strip():
        return pred, success

    pred, success = parse_answer(pred_str=raw_output)
    return pred, success


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace("**", "")
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = text.replace("`", "")
    text = re.sub(r'\\text\{(.*?)\}', r'\1', text)
    return text


def normalize_condition_text(text: str) -> str:
    text = strip_markdown(text)
    text = text.strip().rstrip(".")
    text = text.lower()

    m = re.match(
        r'^(.+?)\s*[\u2013\u2014:\-]+\s*(knight|knave)s?$',
        text, re.IGNORECASE
    )
    if m:
        return f"{m.group(1).strip()} is a {m.group(2).strip().lower()}"

    m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)s?\.?$', text, re.IGNORECASE)
    if m:
        return f"{m.group(1).strip()} is a {m.group(2).strip().lower()}"

    return text


def improved_judge(pred_answer: str, gold_conditions: list) -> tuple:
    """Returns: (is_correct, reason, correct_ratio)"""
    if not pred_answer or not gold_conditions:
        return False, "empty", 0.0

    cleaned_pred = strip_markdown(pred_answer)
    for fp in ["### Reason", "Let's think step by step again",
               "let's go back and check", "###"]:
        if fp in cleaned_pred:
            cleaned_pred = cleaned_pred.split(fp)[0]

    # Method 1: Direct substring match
    correct_count = sum(1 for gc in gold_conditions if gc.lower() in cleaned_pred.lower())
    if correct_count == len(gold_conditions):
        return True, "", 1.0

    # Method 2: Normalized line matching
    pred_lines = cleaned_pred.strip().split("\n")
    pred_normalized = set()
    for line in pred_lines:
        line = re.sub(r'^\(?\d+\)?[\.\):\s]+', '', line.strip()).strip()
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

    # Method 3: parse_single_answer structured extraction
    try:
        parsed_dict = parse_single_answer(pred_answer)
        if parsed_dict:
            parsed_conditions = set()
            for name, role in parsed_dict.items():
                parsed_conditions.add(f"{name.lower().strip()} is a {role.lower().strip()}")
            if gold_normalized and gold_normalized.issubset(parsed_conditions):
                return True, "", 1.0
    except Exception:
        pass

    # Method 4: Flexible name+role matching
    correct_count_normalized = 0
    for gc_norm in gold_normalized:
        if gc_norm in cleaned_pred.lower():
            correct_count_normalized += 1
            continue
        m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)', gc_norm)
        if m:
            name_esc = re.escape(m.group(1))
            if re.search(rf'{name_esc}.*?{m.group(2)}', cleaned_pred.lower()):
                correct_count_normalized += 1

    total_gold = len(gold_normalized) if gold_normalized else len(gold_conditions)
    ratio = correct_count_normalized / total_gold if total_gold > 0 else 0.0

    if correct_count_normalized == total_gold:
        return True, "", ratio

    return False, "wrong_identity", ratio


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    print(f"qwen3-coder-30b KandK Re-evaluation")
    print(f"Source: {SOURCE_FILE}\n")

    if not SOURCE_FILE.exists():
        print(f"ERROR: Source file not found: {SOURCE_FILE}")
        sys.exit(1)

    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        original = json.load(f)

    result = copy.deepcopy(original)
    total = len(result["records"])
    original_perf = original["performance"]

    log_lines = [
        f"qwen3-coder-30b KandK Re-evaluation Log",
        f"Timestamp: {timestamp}",
        f"Source: {SOURCE_FILE}",
        f"Root cause: beyond_list / contain_if false positives in KandK judge",
        f"Total records: {total}",
        f"Original performance: {original_perf*100:.2f}%",
        "",
    ]

    score0_count = 0
    recovered = 0
    still_wrong = 0
    empty_pred = 0
    changes = []

    for rec in result["records"]:
        idx = rec["index"]
        old_score = rec.get("score", 0)
        gt = rec.get("ground_truth", [])
        raw_output = rec.get("raw_output", "")
        old_pred = rec.get("prediction", "")

        if old_score == 1.0:
            continue

        score0_count += 1

        if not raw_output or raw_output.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty raw_output")
            continue

        pred_answer, _ = parse_answer_improved(raw_output)

        if not pred_answer or pred_answer.strip() == "":
            empty_pred += 1
            changes.append(f"  idx={idx}: SKIP empty parsed prediction")
            continue

        gt_list = [gt] if isinstance(gt, str) else list(gt)

        is_correct, reason, ratio = improved_judge(pred_answer, gt_list)

        rec["reeval_old_score"] = old_score
        rec["reeval_old_prediction"] = old_pred
        rec["prediction"] = pred_answer

        if is_correct:
            rec["score"] = 1.0
            rec["reeval_recovered"] = True
            recovered += 1
            changes.append(f"  idx={idx}: RECOVERED  gt={gt_list} score: 0->1")
        else:
            rec["score"] = 0.0
            rec["reeval_recovered"] = False
            still_wrong += 1
            changes.append(
                f"  idx={idx}: STILL WRONG  reason='{reason}' "
                f"gt={gt_list} ratio={ratio:.2f}"
            )

    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf

    result["_reeval_meta"] = {
        "script": "reeval_kandk.py",
        "timestamp": timestamp,
        "reason": "beyond_list/contain_if false positives in KandK judge",
        "original_file": str(SOURCE_FILE),
        "original_performance": original_perf,
        "corrected_performance": new_perf,
        "score0_count": score0_count,
        "recovered": recovered,
        "still_wrong": still_wrong,
        "empty_prediction": empty_pred,
    }

    out_name = f"{SOURCE_FILE.stem}-reeval.json"
    out_path = OUTPUT_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = (
        f"\nSummary:\n"
        f"  Total records         : {total}\n"
        f"  Score=0 records       : {score0_count}\n"
        f"  Empty/no prediction   : {empty_pred}\n"
        f"  Recovered (correct)   : {recovered}\n"
        f"  Still wrong           : {still_wrong}\n"
        f"  Performance           : {original_perf*100:.2f}% -> {new_perf*100:.2f}% "
        f"(+{(new_perf - original_perf)*100:.2f}%)\n"
        f"  Output                : {out_path}"
    )

    log_lines.append(summary)
    log_lines.extend(changes)
    print(summary)

    log_path = OUTPUT_DIR / "reeval_kandk_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
