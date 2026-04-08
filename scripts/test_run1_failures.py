"""Test all 64 'failed' records from run1 through check_correctness to see if they pass now."""
import json
import sys
import multiprocessing
sys.path.insert(0, 'D:/router/LLMRouterBench')

from evaluation.HumanEval.execution import check_correctness


def main():
    with open('D:/router/LLMRouterBench/results/bench/humaneval/test/gpt-5/humaneval-test-gpt-5-20260319_000458.json', encoding='utf-8') as f:
        run1 = json.load(f)

    failed = [r for r in run1['records'] if r.get('score', 0) == 0.0]
    print(f"Testing {len(failed)} failed records from run1...")

    now_pass = 0
    still_fail = 0
    timed_out = 0

    for i, rec in enumerate(failed):
        pred = rec['prediction']
        result = check_correctness(task_id=f'test_{i}', completion_id=0, solution=pred, time_out=5)
        status = result['result']
        if result['passed']:
            now_pass += 1
            print(f"  [{i+1:3d}/{len(failed)}] index={rec['index']:3d} NOW PASSES")
        elif 'timed out' in status:
            timed_out += 1
            print(f"  [{i+1:3d}/{len(failed)}] index={rec['index']:3d} TIMEOUT")
        else:
            still_fail += 1
            print(f"  [{i+1:3d}/{len(failed)}] index={rec['index']:3d} STILL FAILS: {status[:80]}")

    print(f"\nSummary:")
    print(f"  Now passes:  {now_pass}/{len(failed)}")
    print(f"  Still fails: {still_fail}/{len(failed)}")
    print(f"  Timed out:   {timed_out}/{len(failed)}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
