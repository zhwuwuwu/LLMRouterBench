"""
doublecheck.py — 统一重验证脚本

读取原始 JSON 结果文件，用 raw_output 重新 exec 评估每道题，
逐题打印核验结果，将完整结果写入 XX_doublecheck.json（同格式）。

Usage:
    python doublecheck.py                         # 跑所有预设文件
    python doublecheck.py --file <path.json>      # 跑单个文件
    python doublecheck.py --timeout 15            # 自定义超时秒数

使用方法
在你的终端里用 venv 跑：
cd D:\router\LLMRouterBench
# 跑所有 8 个文件（3 HumanEval GPT-5 + 1 HumanEval qwen3 + 3 MBPP GPT-5 + 1 MBPP qwen3）
D:\router\venv_router\Scripts\python.exe doublecheck.py
# 或者只跑单个文件
D:\router\venv_router\Scripts\python.exe doublecheck.py --file "results/bench/humaneval/test/gpt-5/humaneval-test-gpt-5-20260319_011411.json"
# 自定义超时
D:\router\venv_router\Scripts\python.exe doublecheck.py --timeout 15
输出说明
控制台输出 — 每道题都有一行：
  idx  old    new    chg   result
    1   0.0   1.0   0→1 ✓  passed          ← 原来误判，现在通过
   37   0.0   0.0    =     failed: ...      ← 确实失败，不变
  142   1.0   0.0   1→0 ✗  timed out       ← 原来通过，现在超时（如果有）
只有发生变化的题和依然失败的题会逐行打印，纯通过且不变的题不打印（减少刷屏）。
写入文件 — 每个 XX.json 旁边生成 XX_doublecheck.json，格式完全一致，额外增加：
- 每条 record 增加 reeval_detail（执行结果详情）和 reeval_old_score（原始分数）
- 顶层增加 reeval_info 汇总（变化数、gained/lost 等）
- performance 字段更新为重验证后的分数

"""
import json
import sys
import os
import copy
import time
import threading
import ctypes
import argparse
from pathlib import Path
from datetime import datetime

# 确保能导入项目模块
sys.path.insert(0, 'D:/router/LLMRouterBench')

from evaluation.HumanEval.utils import imports as humaneval_imports, sanitize as humaneval_sanitize
from evaluation.MBPP.utils import imports as mbpp_imports, sanitize as mbpp_sanitize

# ─────────────────────────────────────────────────────
# 数据集加载
# ─────────────────────────────────────────────────────

def load_humaneval_dataset():
    """加载 HumanEval 数据集，返回 {0-based index: record}"""
    path = 'D:/router/LLMRouterBench/data/HumanEval/HumanEval.jsonl'
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return {i: d for i, d in enumerate(data)}


def load_mbpp_dataset():
    """加载 MBPP 数据集，返回 {0-based index: record}"""
    path = 'D:/router/LLMRouterBench/data/MBPP/test.json'
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return {i: d for i, d in enumerate(data)}

# ─────────────────────────────────────────────────────
# 代码构建（与原始 evaluator 逻辑完全一致）
# ─────────────────────────────────────────────────────

def build_humaneval_code(raw_output: str, test: str, entry_point: str) -> str:
    extract_code = humaneval_sanitize(raw_output)
    code = "\n".join(humaneval_imports) + "\n" + extract_code + "\n" + test + "\n" + f"check({entry_point})"
    return code


def build_mbpp_code(raw_output: str, test_list: list) -> str:
    extract_code = mbpp_sanitize(raw_output)
    code = "\n".join(mbpp_imports) + "\n" + extract_code + "\n" + "\n".join(test_list)
    return code

# ─────────────────────────────────────────────────────
# 执行引擎（threading + timeout，避免 Windows multiprocessing bug）
# ─────────────────────────────────────────────────────

def exec_with_timeout(code: str, timeout: float = 10.0):
    """
    在线程中 exec 代码，超时后尝试强杀。
    返回 (passed: bool, detail: str)
    """
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

# ─────────────────────────────────────────────────────
# 检测数据集类型
# ─────────────────────────────────────────────────────

def detect_dataset(filepath: str) -> str:
    """从文件路径或 JSON 内容自动检测是 humaneval 还是 mbpp"""
    fname = os.path.basename(filepath).lower()
    if 'humaneval' in fname:
        return 'humaneval'
    elif 'mbpp' in fname:
        return 'mbpp'

    # 从 JSON 内容检测
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    dataset_name = data.get('dataset_name', '').lower()
    if 'humaneval' in dataset_name:
        return 'humaneval'
    elif 'mbpp' in dataset_name:
        return 'mbpp'

    raise ValueError(f"无法识别数据集类型: {filepath}")

# ─────────────────────────────────────────────────────
# 核心：重验证一个 JSON 文件
# ─────────────────────────────────────────────────────

def doublecheck_file(filepath: str, dataset_map: dict, dataset_type: str, timeout: float = 10.0):
    """
    重验证一个结果 JSON 文件。
    - 逐题打印核验结果
    - 返回新的 JSON 数据（同格式，score 已更新）
    """
    with open(filepath, encoding='utf-8') as f:
        run_data = json.load(f)

    # 深拷贝，不修改原数据
    new_data = copy.deepcopy(run_data)
    records = new_data['records']
    total = len(records)

    original_correct = sum(1 for r in records if r.get('score', 0) == 1.0)
    original_perf = run_data.get('performance', 0)

    print(f"\n{'=' * 70}")
    print(f"  File: {os.path.basename(filepath)}")
    print(f"  Dataset: {dataset_type} | Total: {total} | Timeout: {timeout}s")
    print(f"  Original: {original_correct}/{total} ({original_perf:.1%})")
    print(f"{'=' * 70}")
    print()
    print(f"  {'idx':>4s}  {'old':>5s}  {'new':>5s}  {'chg':>5s}  {'result'}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*40}")

    new_correct = 0
    changed_count = 0
    gained = 0
    lost = 0

    for i, rec in enumerate(records):
        idx = rec['index']          # 1-based
        dataset_idx = idx - 1       # 0-based
        raw_output = rec.get('raw_output', '')
        old_score = rec.get('score', 0)

        if dataset_idx not in dataset_map:
            print(f"  {idx:4d}  {old_score:5.1f}    ???    ???  WARNING: index not found in dataset")
            continue

        d = dataset_map[dataset_idx]

        # 构建可执行代码
        if dataset_type == 'humaneval':
            code = build_humaneval_code(raw_output, d['test'], d['entry_point'])
        else:  # mbpp
            code = build_mbpp_code(raw_output, d['test_list'])

        # 执行
        passed, detail = exec_with_timeout(code, timeout=timeout)
        new_score = 1.0 if passed else 0.0

        if new_score == 1.0:
            new_correct += 1

        # 更新 record
        rec['score'] = new_score
        rec['reeval_detail'] = detail
        rec['reeval_old_score'] = old_score

        # 判断变化
        if old_score != new_score:
            changed_count += 1
            if new_score == 1.0:
                gained += 1
                flag = "0→1 ✓"
            else:
                lost += 1
                flag = "1→0 ✗"
            print(f"  {idx:4d}  {old_score:5.1f}  {new_score:5.1f}  {flag:>5s}  {detail[:50]}")
        else:
            status = "pass" if new_score == 1.0 else "fail"
            # 只打印失败的不变项（方便检查）
            if new_score == 0.0:
                print(f"  {idx:4d}  {old_score:5.1f}  {new_score:5.1f}    =   {detail[:50]}")

        # 进度提示
        if (i + 1) % 200 == 0:
            print(f"  ... processed {i+1}/{total}")

    # 更新 performance
    new_perf = new_correct / total if total > 0 else 0.0
    new_data['performance'] = new_perf
    new_data['reeval_info'] = {
        'original_performance': original_perf,
        'original_correct': original_correct,
        'new_correct': new_correct,
        'total': total,
        'changed': changed_count,
        'gained_0_to_1': gained,
        'lost_1_to_0': lost,
        'timeout': timeout,
        'reeval_time': datetime.now().isoformat(),
    }

    # 汇总
    print()
    print(f"  {'─' * 60}")
    print(f"  RESULT: {new_correct}/{total} ({new_perf:.1%})")
    print(f"  Original:  {original_correct}/{total} ({original_perf:.1%})")
    print(f"  Changed:   {changed_count}  (gained 0→1: {gained}, lost 1→0: {lost})")
    print(f"{'=' * 70}")

    return new_data


def write_doublecheck(filepath: str, new_data: dict):
    """写入 _doublecheck.json 文件"""
    p = Path(filepath)
    out_name = p.stem + '_doublecheck' + p.suffix
    out_path = p.parent / out_name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"  → Written to: {out_path}")
    return str(out_path)

# ─────────────────────────────────────────────────────
# 预设的所有文件
# ─────────────────────────────────────────────────────

ALL_FILES = [
    # HumanEval GPT-5 (3 runs)
    'D:/router/LLMRouterBench/results/bench/humaneval/test/gpt-5/humaneval-test-gpt-5-20260319_000458.json',
    'D:/router/LLMRouterBench/results/bench/humaneval/test/gpt-5/humaneval-test-gpt-5-20260319_011330.json',
    'D:/router/LLMRouterBench/results/bench/humaneval/test/gpt-5/humaneval-test-gpt-5-20260319_011411.json',
    # HumanEval qwen3-coder-next
    'D:/router/LLMRouterBench/results/bench/humaneval/test/qwen3-coder-next/humaneval-test-qwen3-coder-next-20260317_012746.json',
    # MBPP GPT-5 (3 runs)
    'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_002829.json',
    'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_023507.json',
    'D:/router/LLMRouterBench/results/bench/mbpp/test/gpt-5/mbpp-test-gpt-5-20260319_024604.json',
    # MBPP qwen3-coder-next
    'D:/router/LLMRouterBench/results/bench/mbpp/test/qwen3-coder-next/mbpp-test-qwen3-coder-next-20260317_050836.json',
]

# ─────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Doublecheck coding benchmark results')
    parser.add_argument('--file', type=str, default=None,
                        help='单个 JSON 文件路径（不指定则跑所有预设文件）')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='每题执行超时秒数（默认 10）')
    args = parser.parse_args()

    files = [args.file] if args.file else ALL_FILES

    # 预加载数据集（只加载需要的）
    need_humaneval = any('humaneval' in f.lower() for f in files)
    need_mbpp = any('mbpp' in f.lower() for f in files)

    humaneval_data = None
    mbpp_data = None

    if need_humaneval:
        humaneval_data = load_humaneval_dataset()
        print(f"Loaded HumanEval dataset: {len(humaneval_data)} problems")
    if need_mbpp:
        mbpp_data = load_mbpp_dataset()
        print(f"Loaded MBPP dataset: {len(mbpp_data)} problems")

    # 逐文件处理
    results_summary = []
    start_all = time.time()

    for filepath in files:
        if not os.path.exists(filepath):
            print(f"\n  ⚠ File not found, skipping: {filepath}")
            continue

        dataset_type = detect_dataset(filepath)
        dataset_map = humaneval_data if dataset_type == 'humaneval' else mbpp_data

        start = time.time()
        new_data = doublecheck_file(filepath, dataset_map, dataset_type, timeout=args.timeout)
        elapsed = time.time() - start

        out_path = write_doublecheck(filepath, new_data)
        info = new_data['reeval_info']
        results_summary.append({
            'file': os.path.basename(filepath),
            'dataset': dataset_type,
            'original': f"{info['original_correct']}/{info['total']} ({info['original_performance']:.1%})",
            'reeval': f"{info['new_correct']}/{info['total']} ({info['new_correct']/info['total']:.1%})",
            'changed': info['changed'],
            'gained': info['gained_0_to_1'],
            'lost': info['lost_1_to_0'],
            'time': f"{elapsed:.1f}s",
        })

    total_elapsed = time.time() - start_all

    # 最终汇总表
    print(f"\n\n{'=' * 90}")
    print(f"  FINAL SUMMARY — {len(results_summary)} files, {total_elapsed:.1f}s total")
    print(f"{'=' * 90}")
    print(f"  {'File':<55s}  {'Original':>12s}  {'Reeval':>12s}  {'Δ':>4s}  {'0→1':>4s}  {'1→0':>4s}")
    print(f"  {'─'*55}  {'─'*12}  {'─'*12}  {'─'*4}  {'─'*4}  {'─'*4}")
    for r in results_summary:
        print(f"  {r['file']:<55s}  {r['original']:>12s}  {r['reeval']:>12s}  {r['changed']:>4d}  {r['gained']:>4d}  {r['lost']:>4d}")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
