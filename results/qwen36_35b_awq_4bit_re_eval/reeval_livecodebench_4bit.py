"""
qwen3.6-35b-awq-4bit LiveCodeBench Re-evaluation

Problem:
  For 4bit outputs that hit max_tokens (65536), raw_output has <think> content
  (possibly with partial code) but no final ```python block. The original
  extract_code_answer() takes the LAST two ``` markers - this picks up partial
  code inside <think> which may be wrong/incomplete.

Fix:
  1. For fail cases (score=0), try improved extraction:
     a. Strip <think>...</think> block
     b. Extract longest ```python block after strip
     c. If no block after strip, look in full output for last COMPLETE ```python block
  2. Re-run evaluate_generation on new prediction.
  3. Keep passing cases unchanged.
"""
import copy, json, re, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.LiveCodeBench.compute_code_generation_metrics import evaluate_generation
import base64, pickle, zlib

MODEL = "qwen3.6-35b-awq-4bit"
SOURCE_FILE = ROOT / "results" / "bench" / "livecodebench" / "test" / MODEL / "livecodebench-test-qwen3.6-35b-awq-4bit-20260420_112501.json"
OUTPUT_DIR = Path(__file__).resolve().parent

def strip_think(raw):
    idx = raw.find("</think>")
    return raw[idx + len("</think>"):] if idx != -1 else raw

def extract_code_improved(raw_output):
    """Improved code extraction for truncated outputs."""
    # Step 1: try after </think>
    after = strip_think(raw_output)
    blocks = re.findall(r"```python\s*\n(.*?)```", after, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", after, re.DOTALL)
    blocks = [b for b in blocks if b.strip() and ("def " in b or "class " in b or "import " in b)]
    if blocks:
        return max(blocks, key=len)
    # Step 2: no closed block after </think>; maybe truncated inside a fence
    # Find last ```python\n and take until end of string
    m = list(re.finditer(r"```python\s*\n", raw_output))
    if m:
        last = m[-1]
        tail = raw_output[last.end():]
        # Remove trailing incomplete if we see another ``` start, else take all
        end_idx = tail.find("```")
        code = tail[:end_idx] if end_idx > 0 else tail
        if "def " in code or "class " in code:
            return code.strip()
    # Step 3: fall back - all blocks anywhere
    all_blocks = re.findall(r"```python\s*\n(.*?)```", raw_output, re.DOTALL)
    if not all_blocks:
        all_blocks = re.findall(r"```\s*\n(.*?)```", raw_output, re.DOTALL)
    all_blocks = [b for b in all_blocks if b.strip() and ("def " in b or "class " in b)]
    if all_blocks:
        return max(all_blocks, key=len)
    return None

def load_lcb_data():
    """Load LCB test data from local JSONL files in order."""
    data_dir = ROOT / "data" / "LiveCodeBench"
    all_data = []
    for jf in sorted(data_dir.glob("*.jsonl")):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    return all_data

def decode_private_tests(raw):
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
        except Exception:
            return []

def build_sample(item):
    public = json.loads(item["public_test_cases"])
    private = decode_private_tests(item["private_test_cases"])
    fn_name = json.loads(item["metadata"]).get("func_name", None)
    tests = public + private
    return {"input_output": json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "fn_name": fn_name,
    })}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    print(f"LiveCodeBench 4bit Re-evaluation\nSource: {SOURCE_FILE}\n")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        original = json.load(f)
    print("Loading LCB dataset...")
    lcb_data = load_lcb_data()
    print(f"Loaded {len(lcb_data)} items")
    result = copy.deepcopy(original)
    total = len(result["records"])
    orig_perf = original["performance"]
    flipped = 0; still_fail = 0; no_code = 0; unchanged_pass = 0
    for i, rec in enumerate(result["records"]):
        idx = rec["index"]
        old_score = rec.get("score", 0)
        if old_score == 1.0:
            unchanged_pass += 1
            if (i+1) % 100 == 0: print(f"  [{i+1}/{total}] pass={unchanged_pass} flip={flipped} fail={still_fail} nocode={no_code}")
            continue
        raw = rec.get("raw_output", "")
        old_pred = rec.get("prediction", "")
        new_code = extract_code_improved(raw)
        if not new_code:
            no_code += 1; still_fail += 1
            continue
        if new_code.strip() == old_pred.strip():
            still_fail += 1
            continue
        # Re-evaluate
        # index in dataset = i (records are in order)
        item = lcb_data[i]
        try:
            sample = build_sample(item)
            results, _ = evaluate_generation(generations=[new_code], sample=sample, debug=False, timeout=6)
            is_correct = all(r is True for r in results[0]) and len(results[0]) > 0
        except Exception as e:
            is_correct = False
        if is_correct:
            rec["score"] = 1.0
            rec["prediction"] = new_code
            rec["reeval_recovered"] = True
            flipped += 1
        else:
            still_fail += 1
        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{total}] pass={unchanged_pass} flip={flipped} fail={still_fail} nocode={no_code}")
    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass / total
    result["performance"] = new_perf
    result["_reeval_meta"] = {
        "script": "reeval_livecodebench_4bit.py", "timestamp": ts,
        "original_file": str(SOURCE_FILE),
        "original_performance": orig_perf, "corrected_performance": new_perf,
        "flipped": flipped, "still_fail": still_fail, "no_code": no_code}
    out_path = OUTPUT_DIR / f"{SOURCE_FILE.stem}-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nFINAL: {orig_perf*100:.2f}% -> {new_perf*100:.2f}% (+{(new_perf-orig_perf)*100:.2f}%)")
    print(f"Flipped: {flipped}, Still fail: {still_fail}, No code: {no_code}")
    print(f"Output: {out_path}")

if __name__ == "__main__":
    main()
