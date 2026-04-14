"""
Re-evaluate GPT-5 ArenaHard results using DeepSeek-V3.2 as judge.

Purpose: The original GPT-5 ArenaHard results were judged by DeepSeek-V3,
but qwen3-coder-next results were judged by DeepSeek-V3.2. This script
re-judges GPT-5's raw outputs using DeepSeek-V3.2 for fair comparison.

Usage:
    cd D:/router/LLMRouterBench
    python scripts/reeval_arenahard_gpt5.py [--concurrency 8] [--dry-run]

Requires .env with:
    ARENA_GRADER_API_KEY=...
    ARENA_GRADER_BASE_URL=http://model-service.aihub.intel.com/v1
    ARENA_GRADER_MODEL_NAME=DeepSeek-V3.2
"""

import json
import os
import re
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI
import httpx

# ============================================================
# Config
# ============================================================

GRADER_MODEL = os.getenv("ARENA_GRADER_MODEL_NAME", "DeepSeek-V3.2")
GRADER_BASE_URL = os.getenv("ARENA_GRADER_BASE_URL", "http://model-service.aihub.intel.com/v1")
GRADER_API_KEY = os.getenv("ARENA_GRADER_API_KEY", "")

GPT5_RESULT_PATH = PROJECT_ROOT / "results/bench/arenahard/gpt-5/arenahard-test-gpt-5-20251029_194015.json"
ARENA_DATA_PATH = PROJECT_ROOT / "data/ArenaHard/arena-hard-v2.jsonl"
BASELINE_DIR = PROJECT_ROOT / "evaluation/ArenaHard/model_answer"

OUTPUT_DIR = PROJECT_ROOT / "results/bench/arenahard/gpt-5-reeval-v3.2"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

# From prompts.py
OG_ARENA_HARD_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

CREATIVE_WRITING_PROMPT = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

When evaluating the assistants' answers, compare both assistants' answers. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is significantly better: [[A>>B]]
2. Assistant A is slightly better: [[A>B]]
3. Tie, relatively the same: [[A=B]]
4. Assistant B is slightly better: [[B>A]]
5. Assistant B is significantly better: [[B>>A]]

Example output: "My final verdict is tie: [[A=B]]"."""

JUDGE_SETTINGS = {
    "hard_prompt": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "coding": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "math": {"baseline": "o3-mini-2025-01-31", "system_prompt": OG_ARENA_HARD_PROMPT},
    "creative_writing": {"baseline": "gemini-2.0-flash-001", "system_prompt": CREATIVE_WRITING_PROMPT},
}

PROMPT_TEMPLATE = """<|User Prompt|>
{QUESTION}

<|The Start of Assistant A's Answer|>
{ANSWER_A}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{ANSWER_B}
<|The End of Assistant B's Answer|>"""

REGEX_PATTERNS = [
    r'\[\[([AB<>=]+)\]\]',
    r'\[([AB<>=]+)\]',
]

# ============================================================
# Core logic
# ============================================================

def create_client():
    """Create OpenAI client for judge model."""
    return OpenAI(
        api_key=GRADER_API_KEY,
        base_url=GRADER_BASE_URL,
        http_client=httpx.Client(verify=False, timeout=600)
    )


def get_score_label(judgment_text: str):
    """Extract score label from judgment text."""
    for pattern_str in REGEX_PATTERNS:
        pattern = re.compile(pattern_str)
        matches = pattern.findall(judgment_text.upper())
        matches = [m for m in matches if m != ""]
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def call_judge(client, system_prompt: str, user_prompt: str, max_retries: int = 5) -> str:
    """Call judge model with retry logic. Returns raw judgment text."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GRADER_MODEL,
                messages=[
                    {"role": "user", "content": f"System: {system_prompt}\n\nUser: {user_prompt}"}
                ],
                temperature=0.0,
                top_p=1.0,
                timeout=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"  [RETRY {attempt+1}/{max_retries}] Judge call failed: {e}, waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Judge call failed after {max_retries} retries")


def calculate_final_score(round1_label, round2_label) -> float:
    """Calculate final score from two rounds of pairwise judgment."""
    round1_map = {"B>>A": 3, "B>A": 1, "A=B": 0, "B=A": 0, "A>B": -1, "A>>B": -3}
    round2_map = {"A>>B": 3, "A>B": 1, "A=B": 0, "B=A": 0, "B>A": -1, "B>>A": -3}

    if round1_label is None or round2_label is None:
        return 0.0

    score1 = round1_map.get(round1_label, 0)
    score2 = round2_map.get(round2_label, 0)
    total = score1 + score2

    if total > 0:
        return 1.0
    elif total == 0:
        return 0.5
    else:
        return 0.0


def load_data():
    """Load all required data."""
    # GPT-5 results
    with open(GPT5_RESULT_PATH, 'r', encoding='utf-8') as f:
        gpt5_data = json.load(f)

    # Arena data (for uid/category mapping)
    arena_data = []
    with open(ARENA_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            arena_data.append(json.loads(line))

    # Baseline answers
    baselines = {}
    for model_file in BASELINE_DIR.glob("*.jsonl"):
        model_name = model_file.stem
        answers = {}
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                answers[entry['uid']] = entry
        baselines[model_name] = answers
        print(f"Loaded {len(answers)} baseline answers for {model_name}")

    return gpt5_data, arena_data, baselines


def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    """Save checkpoint atomically."""
    tmp_path = CHECKPOINT_PATH.with_suffix('.tmp')
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    os.replace(str(tmp_path), str(CHECKPOINT_PATH))


# ============================================================
# Worker
# ============================================================

_lock = threading.Lock()
_completed = 0
_total = 0


def evaluate_one(idx: int, record: dict, arena_entry: dict, baselines: dict, client) -> dict:
    """Evaluate a single record. Returns result dict."""
    global _completed

    uid = arena_entry['uid']
    category = arena_entry['category']
    question = arena_entry['prompt']
    model_answer = record['raw_output']  # GPT-5's raw answer text

    # Get judge settings for this category
    settings = JUDGE_SETTINGS.get(category, JUDGE_SETTINGS['hard_prompt'])
    baseline_model = settings['baseline']
    system_prompt = settings['system_prompt']

    # Get baseline answer
    baseline_entry = baselines.get(baseline_model, {}).get(uid)
    if not baseline_entry:
        print(f"  [WARN] No baseline for uid={uid}, baseline={baseline_model}, scoring 0.0")
        with _lock:
            _completed += 1
        return {
            "index": idx,
            "uid": uid,
            "category": category,
            "subcategory": arena_entry.get('subcategory', ''),
            "old_score": record['score'],
            "new_score": 0.0,
            "round1_label": None,
            "round2_label": None,
        }

    baseline_answer = baseline_entry['messages'][-1]['content']['answer']

    # Round 1: baseline=A, model=B
    user_prompt_r1 = PROMPT_TEMPLATE.format(
        QUESTION=question, ANSWER_A=baseline_answer, ANSWER_B=model_answer
    )
    judgment_r1 = call_judge(client, system_prompt, user_prompt_r1)
    label_r1 = get_score_label(judgment_r1)

    # Round 2: model=A, baseline=B
    user_prompt_r2 = PROMPT_TEMPLATE.format(
        QUESTION=question, ANSWER_A=model_answer, ANSWER_B=baseline_answer
    )
    judgment_r2 = call_judge(client, system_prompt, user_prompt_r2)
    label_r2 = get_score_label(judgment_r2)

    new_score = calculate_final_score(label_r1, label_r2)

    with _lock:
        _completed += 1
        if _completed % 10 == 0 or _completed == _total:
            print(f"  Progress: {_completed}/{_total} ({_completed*100/_total:.1f}%)")

    return {
        "index": idx,
        "uid": uid,
        "category": category,
        "subcategory": arena_entry.get('subcategory', ''),
        "old_score": record['score'],
        "new_score": new_score,
        "round1_label": label_r1,
        "round2_label": label_r2,
        "round1_judgment": judgment_r1,
        "round2_judgment": judgment_r2,
    }


# ============================================================
# Main
# ============================================================

def main():
    global _total, _completed

    parser = argparse.ArgumentParser(description="Re-evaluate GPT-5 ArenaHard with DeepSeek-V3.2")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent judge calls")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be done")
    args = parser.parse_args()

    print(f"=== ArenaHard GPT-5 Re-evaluation ===")
    print(f"Judge model: {GRADER_MODEL}")
    print(f"Judge URL: {GRADER_BASE_URL}")
    print(f"Concurrency: {args.concurrency}")
    print()

    # Load data
    print("Loading data...")
    gpt5_data, arena_data, baselines = load_data()
    records = gpt5_data['records']
    assert len(records) == len(arena_data) == 750, f"Count mismatch: {len(records)} vs {len(arena_data)}"

    # Load checkpoint
    checkpoint = load_checkpoint()
    done_indices = set(int(k) for k in checkpoint.keys())
    print(f"Checkpoint: {len(done_indices)} already completed")

    # Find remaining work
    remaining = [(i, records[i], arena_data[i]) for i in range(750) if i not in done_indices]
    _total = len(remaining)
    _completed = 0

    print(f"Remaining: {_total} records to evaluate ({750 - _total} done)")
    print(f"Total judge API calls needed: {_total * 2}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would evaluate these records:")
        from collections import Counter
        cats = Counter(arena_data[i]['category'] for i, _, _ in remaining)
        subcats = Counter(arena_data[i]['subcategory'] for i, _, _ in remaining)
        print(f"  By category: {dict(cats)}")
        print(f"  By subcategory: {dict(subcats)}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create thread-local clients
    _local = threading.local()

    def get_client():
        if not hasattr(_local, 'client'):
            _local.client = create_client()
        return _local.client

    # Process with thread pool
    start_time = time.time()
    results = dict(checkpoint)  # Start from checkpoint

    checkpoint_interval = 25  # Save every 25 completions
    last_checkpoint_count = len(done_indices)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for idx, record, arena_entry in remaining:
            future = executor.submit(evaluate_one, idx, record, arena_entry, baselines, get_client())
            futures[future] = idx

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[str(idx)] = result

                # Periodic checkpoint
                current_count = len(results)
                if current_count - last_checkpoint_count >= checkpoint_interval:
                    save_checkpoint(results)
                    last_checkpoint_count = current_count

            except Exception as e:
                print(f"  [ERROR] Record {idx} failed: {e}")

    # Final checkpoint
    save_checkpoint(results)

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # ---- Compute summary ----
    all_results = [results[str(i)] for i in range(750) if str(i) in results]

    old_scores = [r['old_score'] for r in all_results]
    new_scores = [r['new_score'] for r in all_results]

    print(f"\n=== Results Summary ===")
    print(f"Total evaluated: {len(all_results)}/750")
    print(f"Old performance (DeepSeek-V3):   {sum(old_scores)/len(old_scores)*100:.2f}%")
    print(f"New performance (DeepSeek-V3.2): {sum(new_scores)/len(new_scores)*100:.2f}%")

    # By subcategory
    from collections import defaultdict
    by_sub = defaultdict(lambda: {"old": [], "new": []})
    for r in all_results:
        sub = r.get('subcategory', 'unknown')
        by_sub[sub]['old'].append(r['old_score'])
        by_sub[sub]['new'].append(r['new_score'])

    print(f"\nBy subcategory:")
    for sub in sorted(by_sub.keys()):
        old = by_sub[sub]['old']
        new = by_sub[sub]['new']
        print(f"  {sub:20s}: old={sum(old)/len(old)*100:.2f}% new={sum(new)/len(new)*100:.2f}% (n={len(old)})")

    # Score change distribution
    changes = {"improved": 0, "degraded": 0, "unchanged": 0}
    for r in all_results:
        if r['new_score'] > r['old_score']:
            changes['improved'] += 1
        elif r['new_score'] < r['old_score']:
            changes['degraded'] += 1
        else:
            changes['unchanged'] += 1
    print(f"\nScore changes: {changes}")

    # ---- Save final result file (same format as original) ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build new records (same structure as original, with updated scores)
    new_records = []
    for i in range(750):
        orig_rec = records[i]
        result = results.get(str(i))
        new_rec = {
            "index": orig_rec['index'],
            "origin_query": orig_rec['origin_query'],
            "prompt": orig_rec['prompt'],
            "prompt_tokens": orig_rec['prompt_tokens'],
            "completion_tokens": orig_rec['completion_tokens'],
            "cost": orig_rec['cost'],
            "score": result['new_score'] if result else orig_rec['score'],
            "prediction": orig_rec['prediction'],
            "ground_truth": orig_rec['ground_truth'],
            "raw_output": orig_rec['raw_output'],
        }
        new_records.append(new_rec)

    new_performance = sum(r['score'] for r in new_records) / len(new_records)
    
    output_data = {
        "performance": new_performance,
        "time_taken": gpt5_data.get('time_taken', 0),
        "prompt_tokens": gpt5_data['prompt_tokens'],
        "completion_tokens": gpt5_data['completion_tokens'],
        "cost": gpt5_data['cost'],
        "counts": 750,
        "model_name": "gpt-5",
        "dataset_name": "arenahard",
        "split": "test",
        "demo": False,
        "data_fingerprint": gpt5_data.get('data_fingerprint', ''),
        "judge_model": GRADER_MODEL,
        "reeval_note": f"Re-evaluated with {GRADER_MODEL} judge (originally judged by DeepSeek-V3)",
        "records": new_records,
    }

    output_path = OUTPUT_DIR / f"arenahard-test-gpt-5-reeval-{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved result to: {output_path}")

    # Also save detailed judgment log
    log_path = OUTPUT_DIR / f"judgments-{timestamp}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved judgment log to: {log_path}")


if __name__ == "__main__":
    main()
