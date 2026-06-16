"""
qwen3.6-35b-awq-4bit MCQ Re-evaluation Script
=============================================

Ported from MiniMax-M2.7 MCQ re-eval logic, with qwen3.6-35b-awq-4bit source
files and output directory, plus truncated <think> handling.
"""

import copy
import json
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "results" / "qwen36_35b_awq_4bit_re_eval"

DATASETS = [
    {
        "name": "arcc",
        "path": ROOT / "results" / "bench" / "arcc" / "test" / "qwen3.6-35b-awq-4bit"
        / "arcc-test-qwen3.6-35b-awq-4bit-20260420_220031.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-E])[.\s\n]?",
    },
    {
        "name": "emorynlp",
        "path": ROOT / "results" / "bench" / "emorynlp" / "test" / "qwen3.6-35b-awq-4bit"
        / "emorynlp-test-qwen3.6-35b-awq-4bit-20260420_222402.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?",
    },
    {
        "name": "gpqa",
        "path": ROOT / "results" / "bench" / "gpqa" / "test" / "qwen3.6-35b-awq-4bit"
        / "gpqa-test-qwen3.6-35b-awq-4bit-20260420_133508.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?",
    },
    {
        "name": "medqa",
        "path": ROOT / "results" / "bench" / "medqa" / "test" / "qwen3.6-35b-awq-4bit"
        / "medqa-test-qwen3.6-35b-awq-4bit-20260420_213713.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-D])[.\s\n]?",
    },
    {
        "name": "meld",
        "path": ROOT / "results" / "bench" / "meld" / "test" / "qwen3.6-35b-awq-4bit"
        / "meld-test-qwen3.6-35b-awq-4bit-20260420_230113.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-G])[.\s\n]?",
    },
    {
        "name": "mmlupro",
        "path": ROOT / "results" / "bench" / "mmlupro" / "test_3000" / "qwen3.6-35b-awq-4bit"
        / "mmlupro-test_3000-qwen3.6-35b-awq-4bit-20260420_174245.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-J])[.\s\n]?",
    },
    {
        "name": "winogrande",
        "path": ROOT / "results" / "bench" / "winogrande" / "valid" / "qwen3.6-35b-awq-4bit"
        / "winogrande-valid-qwen3.6-35b-awq-4bit-20260420_184915.json",
        "answer_pattern": r"(?i)Answer\s*:\s*\$?([A-B])[.\s\n]?",
    },
]


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning block; handle truncated think-only output."""
    if not text:
        return ""
    stripped = text.lstrip()
    if stripped.startswith("<think>") and "</think>" not in stripped:
        return ""
    idx = text.find("</think>")
    if idx != -1:
        return text[idx + len("</think>"):]
    return text


def extract_answer_fixed(text: str, answer_pattern: str) -> str:
    if not text:
        return ""

    cleaned = strip_think(text)
    if not cleaned:
        return ""

    cleaned = re.sub(r'\*{1,3}\s*(Answer)\s*\*{1,3}', r'\1', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r':\s*\*{1,3}\s*', ': ', cleaned)
    cleaned = re.sub(r'(\b[A-J])\*{1,3}', r'\1', cleaned)

    matches = re.findall(answer_pattern, cleaned)
    if matches:
        return matches[-1].strip()

    final_pattern = answer_pattern.replace(r"Answer\s*:\s", r"Final Answer\s\n+\s")
    final_match = re.search(final_pattern, cleaned)
    if final_match:
        return final_match.group(1).strip()

    return ""


def reeval_dataset(config: dict, log_lines: list[str]) -> dict:
    name = config["name"]
    result_path = config["path"]
    answer_pattern = config["answer_pattern"]

    log_lines.append(f"\n{'=' * 60}")
    log_lines.append(f"Dataset: {name}")
    log_lines.append(f"Source : {result_path}")
    log_lines.append(f"Pattern: {answer_pattern}")
    log_lines.append(f"{'=' * 60}")

    with open(result_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    result = copy.deepcopy(original)
    total = len(result["records"])
    original_perf = original["performance"]

    empty_pred_count = 0
    recovered = 0
    still_empty = 0
    wrong_answer = 0
    skipped_gen_fail = 0

    changes = []

    for rec in result["records"]:
        idx = rec["index"]
        old_pred = rec.get("prediction", "")
        old_score = rec.get("score", 0)
        raw = rec.get("raw_output", "")
        gt = rec.get("ground_truth", "")

        if old_pred != "" or old_score == 1.0:
            continue

        if not raw or raw.strip() == "":
            skipped_gen_fail += 1
            changes.append(f"  idx={idx}: skipped empty raw_output")
            continue

        empty_pred_count += 1
        new_pred = extract_answer_fixed(raw, answer_pattern)

        if not new_pred:
            still_empty += 1
            changes.append(f"  idx={idx}: still empty after fix")
            continue

        rec["reeval_old_prediction"] = old_pred
        rec["reeval_old_score"] = old_score

        if new_pred == gt:
            rec["prediction"] = new_pred
            rec["score"] = 1.0
            recovered += 1
            changes.append(f"  idx={idx}: RECOVERED pred='{new_pred}' gt='{gt}' score: 0->1")
        else:
            rec["prediction"] = new_pred
            rec["score"] = 0.0
            wrong_answer += 1
            changes.append(f"  idx={idx}: EXTRACTED but wrong pred='{new_pred}' gt='{gt}' score: 0->0")

    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total if total > 0 else 0.0
    result["performance"] = new_perf

    result["_reeval_meta"] = {
        "script": "reeval_mcq_4bit.py",
        "timestamp": datetime.now().isoformat(),
        "reason": "ANSWER_PATTERN regex fails on markdown answer formatting and truncated thinking-model output",
        "original_file": str(result_path),
        "original_performance": original_perf,
        "corrected_performance": new_perf,
        "empty_pred_found": empty_pred_count,
        "recovered": recovered,
        "still_empty": still_empty,
        "wrong_answer_extracted": wrong_answer,
        "skipped_gen_fail": skipped_gen_fail,
    }

    out_path = OUTPUT_DIR / f"{result_path.stem}-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = (
        f"\n  Summary for {name}:\n"
        f"    Total records      : {total}\n"
        f"    Empty predictions  : {empty_pred_count}\n"
        f"    Recovered (correct): {recovered}\n"
        f"    Extracted (wrong)  : {wrong_answer}\n"
        f"    Still empty        : {still_empty}\n"
        f"    Skipped (gen fail) : {skipped_gen_fail}\n"
        f"    Performance        : {original_perf * 100:.2f}% -> {new_perf * 100:.2f}% "
        f"(+{(new_perf - original_perf) * 100:.2f}%)\n"
        f"    Output             : {out_path}"
    )
    log_lines.append(summary)
    log_lines.extend(changes)

    print(
        f"  {name:12s}: {empty_pred_count} empty -> {recovered} recovered, "
        f"{wrong_answer} wrong, {still_empty} still empty | "
        f"{original_perf * 100:.2f}% -> {new_perf * 100:.2f}%"
    )

    return {
        "name": name,
        "total": total,
        "empty_pred": empty_pred_count,
        "recovered": recovered,
        "wrong_answer": wrong_answer,
        "still_empty": still_empty,
        "skipped_gen_fail": skipped_gen_fail,
        "original_perf": original_perf,
        "new_perf": new_perf,
        "output_file": str(out_path),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("qwen3.6-35b-awq-4bit MCQ Re-evaluation")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"Datasets  : {len(DATASETS)}\n")

    log_lines = [
        "qwen3.6-35b-awq-4bit MCQ Re-evaluation Log",
        f"Timestamp: {datetime.now().isoformat()}",
        "Root cause: ANSWER_PATTERN regex misses markdown answer formatting and truncated think-block output",
        "Fix: strip think block and markdown markers before regex extraction",
    ]

    all_summaries = []
    total_recovered = 0
    total_empty = 0
    total_wrong = 0
    total_still_empty = 0
    total_skipped = 0

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
        total_skipped += summary["skipped_gen_fail"]

    grand = (
        f"\n\n{'=' * 60}\n"
        f"GRAND SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Total empty predictions found : {total_empty}\n"
        f"  Total recovered (correct)     : {total_recovered}\n"
        f"  Total extracted (wrong answer): {total_wrong}\n"
        f"  Total still empty             : {total_still_empty}\n"
        f"  Total skipped (gen fail)      : {total_skipped}\n"
        f"\n  Per-dataset performance changes:"
    )
    log_lines.append(grand)

    print(f"\n{'=' * 60}")
    print("GRAND SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  Total empty: {total_empty} -> Recovered: {total_recovered}, "
        f"Wrong: {total_wrong}, Still empty: {total_still_empty}, Skipped: {total_skipped}"
    )

    for s in all_summaries:
        line = (
            f"    {s['name']:12s}: {s['original_perf'] * 100:.2f}% -> "
            f"{s['new_perf'] * 100:.2f}% (+{(s['new_perf'] - s['original_perf']) * 100:.2f}%)"
        )
        log_lines.append(line)
        print(line)

    log_path = OUTPUT_DIR / "reeval_mcq_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    main()
