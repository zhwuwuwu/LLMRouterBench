"""
Re-evaluate qwen3-coder-next HumanEval run.
Uses direct exec() instead of multiprocessing to avoid Windows spawn overhead.
Sequential execution, 10s timeout via threading.
"""
import json
import sys
import time
import threading
import ctypes

sys.path.insert(0, 'D:/router/LLMRouterBench')

from evaluation.HumanEval.utils import imports, sanitize


def build_prediction(raw_output: str, test: str, entry_point: str) -> str:
    """Replicate HumanEvalEvaluator.extract_code_answer logic."""
    extract_code = sanitize(raw_output)
    code = "\n".join(imports) + "\n" + extract_code + "\n" + test + "\n" + f"check({entry_point})"
    return code


def exec_with_timeout(code: str, timeout: float = 10.0):
    """Execute code with timeout using a thread. Returns (passed, detail)."""
    result = [None]

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

    for rec in records:
        idx = rec['index']  # 1-based
        dataset_idx = idx - 1  # 0-based
        raw_output = rec.get('raw_output', '')
        old_score = rec.get('score', 0)

        if dataset_idx not in dataset_map:
            print(f"  WARNING: index {idx} not found in dataset")
            continue

        d = dataset_map[dataset_idx]
        test = d['test']
        entry_point = d['entry_point']

        prediction = build_prediction(raw_output, test, entry_point)
        passed, detail = exec_with_timeout(prediction, timeout=timeout)

        new_score = 1.0 if passed else 0.0
        if new_score == 1.0:
            new_correct += 1

        if old_score != new_score:
            direction = "0->1" if new_score == 1.0 else "1->0"
            changed.append((idx, old_score, new_score, direction, detail[:80]))
            print(f"  CHANGED idx={idx:3d}: {direction}  ({detail[:80]})")

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
    print("=" * 60)
    return new_correct, total, changed


def main():
    from evaluation.base_evaluator import BaseEvaluator

    class Loader(BaseEvaluator):
        def __init__(self): super().__init__()
        def load_data(self, split="test"): return []
        def get_valid_splits(self): return ["test"]
        def evaluate(self, data, output_text, **kw): return {}

    loader = Loader()
    raw_data = loader.load_jsonl('data/HumanEval/HumanEval.jsonl')
    print(f"Loaded {len(raw_data)} HumanEval problems\n")

    runs = [
        ('qwen3-coder-next', 'D:/router/LLMRouterBench/results/bench/humaneval/test/qwen3-coder-next/humaneval-test-qwen3-coder-next-20260317_012746.json'),
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
