"""
MiniMax-M2.7 HumanEval / MBPP Re-evaluation Script
====================================================

Problem:
  The original sanitize() code extractor fails on MiniMax-M2.7 output because
  the model wraps its answer in <think>...</think> reasoning tags followed by
  extensive markdown commentary.  sanitize() → extract_longest_valid_code()
  treats the *entire* text (markdown + code) as raw Python, truncates at 100
  lines, and frequently drops or corrupts the actual function definitions.

Fix:
  1. Strip everything up to and including </think>.
  2. Extract ```python ... ``` (or bare ```) fenced code blocks.
  3. Pick the longest block (the complete solution).
  4. Fall back to the original sanitize() path if no fenced block is found.
  5. Re-run check_correctness one-by-one (sequential) to avoid the Windows
     multiprocessing contention issue we saw with GPT-5.

Outputs:
  Copies of the original result JSONs with corrected score / prediction /
  performance fields, written to results/minimax_reeval/.
"""

import copy
import json
import os
import re
import sys
import time
import multiprocessing
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths (relative to LLMRouterBench root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]          # LLMRouterBench/
sys.path.insert(0, str(ROOT))

HUMANEVAL_RESULT = (
    ROOT / "results" / "bench" / "humaneval" / "test" / "MiniMax-M2.7"
    / "humaneval-test-MiniMax-M2.7-20260331_044221.json"
)
MBPP_RESULT = (
    ROOT / "results" / "bench" / "mbpp" / "test" / "MiniMax-M2.7"
    / "mbpp-test-MiniMax-M2.7-20260409_180939.json"
)

HUMANEVAL_DATA = ROOT / "data" / "HumanEval" / "HumanEval.jsonl"
MBPP_DATA      = ROOT / "data" / "MBPP" / "test.json"

OUTPUT_DIR = ROOT / "results" / "minimax_reeval"

# ---------------------------------------------------------------------------
# Imports list (same as evaluation/HumanEval/utils.py)
# ---------------------------------------------------------------------------
IMPORTS = [
    "import math", "import re", "import sys", "import copy",
    "import datetime", "import itertools", "import collections",
    "import heapq", "import functools", "import hashlib",
    "import numpy", "import numpy as np", "import string",
    "from typing import *", "from collections import *",
]
IMPORT_BLOCK = "\n".join(IMPORTS)

# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

def strip_think(raw: str) -> str:
    """Remove <think>...</think> block from model output."""
    idx = raw.find("</think>")
    if idx != -1:
        return raw[idx + len("</think>"):]
    return raw


def extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks (```python or bare ```)."""
    # Try ```python first, then bare ```
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return [b for b in blocks if b.strip()]


def extract_code_from_raw(raw_output: str) -> str | None:
    """
    Extract the solution code from raw_output.
    Returns the longest fenced code block after stripping <think>,
    or None if no block is found.
    """
    after_think = strip_think(raw_output)
    blocks = extract_code_blocks(after_think)
    if not blocks:
        return None
    return max(blocks, key=len)

# ---------------------------------------------------------------------------
# Execution (copied from evaluation/HumanEval/execution.py, sequential only)
# ---------------------------------------------------------------------------

import io
import signal
import threading
import tempfile
import platform
import contextlib
import faulthandler


def unsafe_execute(result: list, solution: str, time_out: float):
    with _create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os as _os
        import shutil as _shutil
        _rmtree = _shutil.rmtree
        _rmdir  = _os.rmdir
        _chdir  = _os.chdir

        # Pre-import heavy modules BEFORE reliability_guard() disables os.putenv etc.
        # numpy's __init__ calls os.putenv() which guard sets to None.
        # By importing here, numpy is cached in sys.modules and won't re-init during exec.
        try:
            import numpy  # noqa: F401
        except ImportError:
            pass

        _reliability_guard()

        try:
            exec_globals = {}
            with _swallow_io():
                with _time_limit(time_out):
                    exec(solution, exec_globals)
            result.append("passed")
        except _TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")

        # Restore before tempdir context manager exits (needs os.chdir for cleanup).
        _shutil.rmtree = _rmtree
        _os.rmdir = _rmdir
        _os.chdir = _chdir


def check_correctness_sequential(solution: str, time_out: float = 5, max_retries: int = 2) -> dict:
    """
    Run a single test in a child process (sequential).
    Retries up to max_retries times on non-assertion failures (e.g. flaky
    timeout / NoneType errors from Windows multiprocessing contention).
    """
    for attempt in range(max_retries):
        manager = multiprocessing.Manager()
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute, args=(result, solution, time_out))
        p.start()
        p.join(timeout=time_out + 2)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)

        if not result:
            result.append("timed out")

        verdict = result[0]
        # If passed or a genuine assertion/logic error, accept immediately.
        # Only retry on flaky infra errors (timed out, NoneType, Manager issues).
        if verdict == "passed":
            break
        if verdict.startswith("failed: ") and "NoneType" not in verdict and "timed out" not in verdict:
            break
        # Otherwise retry (flaky)
        if attempt < max_retries - 1:
            time.sleep(0.2)

    return {
        "passed": result[0] == "passed",
        "result": result[0],
    }


# -- helpers (same as execution.py) --

class _TimeoutException(Exception):
    pass

@contextlib.contextmanager
def _time_limit(seconds: float):
    if platform.system() != "Windows":
        def handler(signum, frame):
            raise _TimeoutException("Timed out!")
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    else:
        timer = threading.Timer(seconds, lambda: None)
        timer.start()
        try:
            yield
        finally:
            timer.cancel()

@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield

class _WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **k):   raise IOError
    def readline(self, *a, **k): raise IOError
    def readlines(self, *a, **k): raise IOError
    def readable(self, *a, **k): return False

class _redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"

@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as d:
        with _chdir_ctx(d):
            yield d

@contextlib.contextmanager
def _chdir_ctx(root):
    if root == ".":
        yield; return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)

def _reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes,)*2)
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes,)*2)
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes,)*2)
    faulthandler.disable()
    import builtins
    builtins.exit = None; builtins.quit = None
    os.environ["OMP_NUM_THREADS"] = "1"
    for attr in [
        "kill","system","putenv","remove","removedirs","rmdir","fchdir",
        "setuid","fork","forkpty","killpg","rename","renames","truncate",
        "replace","unlink","fchmod","fchown","chmod","chown","chroot",
        "lchflags","lchmod","lchown","getcwd","chdir",
    ]:
        if hasattr(os, attr):
            setattr(os, attr, None)
    import shutil
    shutil.rmtree = None; shutil.move = None; shutil.chown = None
    import subprocess
    subprocess.Popen = None
    __builtins__["help"] = None
    for mod in ["ipdb","joblib","resource","psutil","tkinter"]:
        sys.modules[mod] = None

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_humaneval_data() -> dict:
    """Returns dict: task_id -> item.  Items have 'test', 'entry_point'."""
    data = {}
    with open(HUMANEVAL_DATA, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            data[item["task_id"]] = item
    return data


def load_mbpp_data() -> list[dict]:
    """Returns list of items in file order (same order as result records)."""
    items = []
    with open(MBPP_DATA, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line.strip()))
    return items

# ---------------------------------------------------------------------------
# Re-evaluation logic
# ---------------------------------------------------------------------------

def reeval_humaneval():
    print("=" * 60)
    print("Re-evaluating MiniMax-M2.7 HumanEval")
    print("=" * 60)

    with open(HUMANEVAL_RESULT, "r", encoding="utf-8") as f:
        original = json.load(f)

    he_data = load_humaneval_data()
    result = copy.deepcopy(original)

    total = len(result["records"])
    old_pass = sum(1 for r in result["records"] if r["score"] == 1.0)
    flipped = 0
    still_fail = 0
    already_pass = 0

    for i, rec in enumerate(result["records"]):
        idx = rec["index"]                         # 1-based
        task_id = f"HumanEval/{idx - 1}"
        he_item = he_data[task_id]
        test_code = he_item["test"]
        entry_point = he_item["entry_point"]

        if rec["score"] == 1.0:
            already_pass += 1
            _progress(i + 1, total, f"#{idx} already pass")
            continue

        # --- Try improved extraction ---
        code_block = extract_code_from_raw(rec["raw_output"])
        if code_block is None:
            still_fail += 1
            _progress(i + 1, total, f"#{idx} no code block → still fail")
            continue

        solution = IMPORT_BLOCK + "\n" + code_block + "\n" + test_code + "\n" + f"check({entry_point})"
        verdict = check_correctness_sequential(solution, time_out=5)

        if verdict["passed"]:
            rec["score"] = 1.0
            rec["prediction"] = IMPORT_BLOCK + "\n" + code_block + "\n" + test_code + "\n" + f"check({entry_point})"
            flipped += 1
            _progress(i + 1, total, f"#{idx} FLIPPED → pass")
        else:
            still_fail += 1
            _progress(i + 1, total, f"#{idx} still fail ({verdict['result'][:40]})")

    new_pass = sum(1 for r in result["records"] if r["score"] == 1.0)
    result["performance"] = new_pass / total
    result["_reeval_meta"] = {
        "script": "reeval_minimax_coding.py",
        "timestamp": datetime.now().isoformat(),
        "reason": "sanitize code extraction failed on <think>+markdown output",
        "original_file": str(HUMANEVAL_RESULT),
        "original_performance": original["performance"],
        "corrected_performance": result["performance"],
        "flipped": flipped,
        "still_fail": still_fail,
    }

    out_path = OUTPUT_DIR / "humaneval-test-MiniMax-M2.7-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n\n--- HumanEval Summary ---")
    print(f"  Original : {old_pass}/{total} = {original['performance']*100:.2f}%")
    print(f"  Corrected: {new_pass}/{total} = {result['performance']*100:.2f}%")
    print(f"  Flipped  : {flipped}")
    print(f"  Still fail: {still_fail}")
    print(f"  Output   : {out_path}\n")


def reeval_mbpp():
    print("=" * 60)
    print("Re-evaluating MiniMax-M2.7 MBPP")
    print("=" * 60)

    with open(MBPP_RESULT, "r", encoding="utf-8") as f:
        original = json.load(f)

    mbpp_items = load_mbpp_data()
    result = copy.deepcopy(original)

    total = len(result["records"])
    old_pass = sum(1 for r in result["records"] if r["score"] == 1.0)
    flipped = 0
    still_fail = 0
    already_pass = 0

    for i, rec in enumerate(result["records"]):
        idx = rec["index"]  # 1-based
        mbpp_item = mbpp_items[i]  # same order as result records

        if rec["score"] == 1.0:
            already_pass += 1
            _progress(i + 1, total, f"#{idx} already pass")
            continue

        # --- Try improved extraction ---
        code_block = extract_code_from_raw(rec["raw_output"])
        if code_block is None:
            still_fail += 1
            _progress(i + 1, total, f"#{idx} no code block → still fail")
            continue

        test_list = mbpp_item["test_list"]
        solution = IMPORT_BLOCK + "\n" + code_block + "\n" + "\n".join(test_list)
        verdict = check_correctness_sequential(solution, time_out=5)

        if verdict["passed"]:
            rec["score"] = 1.0
            rec["prediction"] = IMPORT_BLOCK + "\n" + code_block + "\n" + "\n".join(test_list)
            flipped += 1
            _progress(i + 1, total, f"#{idx} FLIPPED → pass")
        else:
            still_fail += 1
            _progress(i + 1, total, f"#{idx} still fail ({verdict['result'][:40]})")

    new_pass = sum(1 for r in result["records"] if r["score"] == 1.0)
    result["performance"] = new_pass / total
    result["_reeval_meta"] = {
        "script": "reeval_minimax_coding.py",
        "timestamp": datetime.now().isoformat(),
        "reason": "sanitize code extraction failed on <think>+markdown output",
        "original_file": str(MBPP_RESULT),
        "original_performance": original["performance"],
        "corrected_performance": result["performance"],
        "flipped": flipped,
        "still_fail": still_fail,
    }

    out_path = OUTPUT_DIR / "mbpp-test-MiniMax-M2.7-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n\n--- MBPP Summary ---")
    print(f"  Original : {old_pass}/{total} = {original['performance']*100:.2f}%")
    print(f"  Corrected: {new_pass}/{total} = {result['performance']*100:.2f}%")
    print(f"  Flipped  : {flipped}")
    print(f"  Still fail: {still_fail}")
    print(f"  Output   : {out_path}\n")


def _progress(cur, total, msg):
    pct = cur / total * 100
    print(f"\r  [{cur:>4}/{total}] {pct:5.1f}%  {msg:<60}", end="", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Timestamp : {datetime.now().isoformat()}\n")

    reeval_humaneval()
    reeval_mbpp()

    print("=" * 60)
    print("Done. All re-evaluated results saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)
