"""Compute token/time/cost stats for qwen3.6-35b-awq-4bit across 17 datasets.

Uses reeval accuracy for humaneval/mbpp/kandk/livecodebench.
Tokens/time from original bench files.
"""
import json
import os
from pathlib import Path

BENCH = Path("D:/router/LLMRouterBench/results/bench")
REEVAL = Path("D:/router/LLMRouterBench/results/qwen36_35b_awq_4bit_re_eval")
MODEL = "qwen3.6-35b-awq-4bit"

# Dataset -> (split, domain)
DATASETS = [
    ("aime", "hybrid", "Math"),
    ("math500", "test", "Math"),
    ("mathbench", "test", "Math"),
    ("livemathbench", "test", "Math"),
    ("humaneval", "test", "Coding"),
    ("mbpp", "test", "Coding"),
    ("livecodebench", "test", "Coding"),
    ("bbh", "test", "Logic"),
    ("kandk", "test", "Logic"),
    ("mmlupro", "test_3000", "Knowledge"),
    ("gpqa", "test", "Knowledge"),
    ("finqa", "test", "Knowledge"),
    ("medqa", "test", "Knowledge"),
    ("arcc", "test", "Knowledge"),
    ("winogrande", "valid", "Knowledge"),
    ("emorynlp", "test", "Affective"),
    ("meld", "test", "Affective"),
]

# Reeval accuracy overrides (fraction of correct / total)
REEVAL_FILES = {
    "humaneval": REEVAL / "humaneval-test-qwen3.6-35b-awq-4bit-reeval.json",
    "mbpp": REEVAL / "mbpp-test-qwen3.6-35b-awq-4bit-reeval.json",
    "kandk": REEVAL / "kandk-test-qwen3.6-35b-awq-4bit-20260420_193019-reeval.json",
    "livecodebench": REEVAL / "livecodebench-test-qwen3.6-35b-awq-4bit-20260420_112501-reeval.json",
}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def stats_for(dataset, split):
    ddir = BENCH / dataset / split / MODEL
    files = sorted(ddir.glob("*.json"))
    assert files, f"no files in {ddir}"
    # Use the non-reeval file for token/time stats
    orig = [f for f in files if "reeval" not in f.name.lower()][0]
    data = load_json(orig)
    records = data if isinstance(data, list) else data.get("results", data.get("records", []))
    pt = ct = tt = 0
    total = 0
    correct = 0
    scores_none = 0
    for r in records:
        total += 1
        s = r.get("score")
        if s is None:
            scores_none += 1
        elif s >= 0.5:
            correct += 1
        pt += r.get("prompt_tokens", 0) or 0
        ct += r.get("completion_tokens", 0) or 0
        tt += r.get("processing_time", 0) or 0
    acc = correct / total * 100 if total else 0
    # Override with reeval if available
    if dataset in REEVAL_FILES:
        rdata = load_json(REEVAL_FILES[dataset])
        rrecs = rdata if isinstance(rdata, list) else rdata.get("results", rdata.get("records", []))
        # Skip meta entries
        rrecs = [r for r in rrecs if isinstance(r, dict) and "score" in r]
        rc = sum(1 for r in rrecs if (r.get("score") or 0) >= 0.5)
        rt = len(rrecs)
        if rt:
            acc = rc / rt * 100
    cost = (pt + ct) / 1_000_000 * 0.01  # symbolic $0.01/1M
    return {
        "samples": total,
        "acc": acc,
        "pt": pt,
        "ct": ct,
        "tt": tt,
        "cost": cost,
    }


def main():
    rows = []
    total_pt = total_ct = total_tt = 0
    total_cost = 0
    total_samples = 0
    acc_sum = 0
    for ds, split, domain in DATASETS:
        s = stats_for(ds, split)
        rows.append((ds, domain, s))
        total_pt += s["pt"]
        total_ct += s["ct"]
        total_tt += s["tt"]
        total_cost += s["cost"]
        total_samples += s["samples"]
        acc_sum += s["acc"]
    print(f"{'Dataset':<16}{'Domain':<12}{'N':>6}{'Acc':>8}{'PT':>12}{'CT':>14}{'Time(s)':>12}{'Cost($)':>10}")
    for ds, domain, s in rows:
        print(f"{ds:<16}{domain:<12}{s['samples']:>6}{s['acc']:>7.2f}%{s['pt']:>12,}{s['ct']:>14,}{s['tt']:>12.0f}{s['cost']:>10.4f}")
    print(f"{'TOTAL':<16}{'':<12}{total_samples:>6}{acc_sum/len(rows):>7.2f}%{total_pt:>12,}{total_ct:>14,}{total_tt:>12.0f}{total_cost:>10.4f}")
    print(f"\nAVG accuracy: {acc_sum/len(rows):.2f}%")
    # Domain avg
    by_domain = {}
    for ds, dom, s in rows:
        by_domain.setdefault(dom, []).append(s["acc"])
    print("\nDomain Averages:")
    for dom, accs in by_domain.items():
        print(f"  {dom}: {sum(accs)/len(accs):.2f}% (n={len(accs)})")


if __name__ == "__main__":
    main()
