"""
Re-evaluate all 3 GPT-5 MBPP runs.
Uses direct exec() instead of multiprocessing to avoid Windows spawn overhead.
Sequential execution, 10s timeout via threading.
"""
import json
import sys
import time
import threading
import ctypes

sys.path.insert(0, 'D:/router/LLMRouterBench')

from evaluation.MBPP.utils import imports, sanitize


def build_prediction(raw_output: str, test_list: list) -> str:
    """Replicate MBPPEvaluator.extract_code_answer logic."""
    extract_code = sanitize(raw_output)
    code = "\n".join(imports) + "\n" + extract_code + "\n" + "\n".join(test_list)
    return code


def exec_with_timeout(code: str, timeout: float = 10.0):
    """Execute code with timeout using a thread. Returns (passed, detail)."""
    result = [None]  # [None] = not finished, ["passed"], ["failed: ..."], ["timed out"]

    def run():
        try:
            exec_globals = {}
            exec(code, exec_globals)
            result[0] = "passed"
        except Exception as e:
            result[0] = f"failed: {e}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result[0] is None:
        result[0] = "timed out"
        # Try to kill the thread (best effort on Windows)
        try:
            if t.is_alive() and hasattr(ctypes, 'pythonapi'):
                tid = t.ident
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(tid),
                    ctypes.py_object(SystemExit)
                )
        except Exception:
            pass

    return result[0] == "passed", result[0]


def re_evaluate_run(filepath: str, raw_data: list, timeout: float = 10.0):
    """Re-evaluate all records in a run file sequentially."""
    with open(filepath, encoding='utf-8') as f:
        run_data = json.load(f)

    # MBPP dataset is indexed by position in list (0-based), records use 1-based index
    dataset_map = {i: d for i, d in enumerate(raw_data)}

    records = run_data['records']
    total = len(records)
    original_correct = sum(1 for r in records if r.get('score', 0) == 1.0)
    original_perf = run_data.get('performance', 0)

    print(f"File: {filepath.split('/')[-1]}")
    print(f"Original: {original_correct}/{total} correct ({original_perf:.1%})")
    print(f"Re-evaluating sequentially with timeout={timeout}s...")
    print()

    new_correct = 0
    changed = []
    timed_out_count = 0
    failed_count = 0

    for i, rec in enumerate(records):
        idx = rec['index']  # 1-based
        dataset_idx = idx - 1  # 0-based
        raw_output = rec.get('raw_output', '')
        old_score = rec.get('score', 0)

        if dataset_idx not in dataset_map:
            print(f"  WARNING: index {idx} not found in dataset")
            continue

        d = dataset_map[dataset_idx]
        test_list = d['test_list']

        prediction = build_prediction(raw_output, test_list)
        passed, detail = exec_with_timeout(prediction, timeout=timeout)

        new_score = 1.0 if passed else 0.0
        if new_score == 1.0:
            new_correct += 1
        else:
            if "timed out" in detail:
                timed_out_count += 1
            else:
                failed_count += 1

        if old_score != new_score:
            direction = "0->1" if new_score == 1.0 else "1->0"
            changed.append((idx, old_score, new_score, direction, detail[:80]))
            if len(changed) <= 30:  # Only print first 30 changes to avoid flooding
                print(f"  CHANGED idx={idx:3d}: {direction}  ({detail[:80]})")

        # Progress indicator every 100 records
        if (i + 1) % 100 == 0:
            print(f"  ... processed {i+1}/{total}")

    new_perf = new_correct / total
    print()
    print(f"Result: {new_correct}/{total} correct ({new_perf:.1%})")
    print(f"  Original: {original_correct}/{total} ({original_perf:.1%})")
    print(f"  Changed:  {len(changed)} records")
    if changed:
        gained = sum(1 for c in changed if c[3] == "0->1")
        lost = sum(1 for c in changed if c[3] == "1->0")
        print(f"    Gained (0->1): {gained}")
        print(f"    Lost   (1->0): {lost}")
    print(f"  Still failing: {failed_count} errors, {timed_out_count} timeouts")
    print("=" * 60)
    return new_correct, total, changed


def main():
    # Load MBPP dataset
    import json as _json
    data_path = 'D:/router/LLMRouterBench/data/MBPP/test.json'
    raw_data = []
    with open(data_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(_json.loads(line))
    print(f"Loaded {len(raw_data)} MBPP problems\n")

    runs = [
        ('run1', 'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_002829.json'),
        ('run2', 'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_023507.json'),
        ('run3', 'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_024604.json'),
    ]

    all_results = {}
    for name, path in runs:
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")
        correct, total, changed = re_evaluate_run(path, raw_data, timeout=10.0)
        all_results[name] = (correct, total, changed)

    print(f"\n\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}")
    for name, (correct, total, changed) in all_results.items():
        print(f"  {name}: {correct}/{total} ({correct/total:.1%})")


if __name__ == '__main__':
    main()
