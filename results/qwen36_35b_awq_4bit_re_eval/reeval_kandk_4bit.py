"""qwen3.6-35b-awq-4bit KandK Re-evaluation (adapted from reeval_kandk.py)"""
import copy, json, re, sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluation.K_and_K.scoring import parse_answer, parse_single_answer

MODEL = "qwen3.6-35b-awq-4bit"
SOURCE_FILE = ROOT / "results" / "bench" / "kandk" / "test" / MODEL / "kandk-test-qwen3.6-35b-awq-4bit-20260420_193019.json"

def strip_think(raw):
    idx = raw.find("</think>")
    return raw[idx+len("</think>"):] if idx != -1 else raw

def parse_answer_improved(raw_output):
    text = strip_think(raw_output)
    text_n = re.sub(r'#{1,3}\s*CONCLUSION\s*[:]*\s*\*{0,3}\s*', 'CONCLUSION:', text, flags=re.IGNORECASE)
    text_n = re.sub(r'CONCLUSION\s*[:]*\s*\*{1,3}\s*', 'CONCLUSION:', text_n, flags=re.IGNORECASE)
    best = None
    for pat in ['CONCLUSION:', 'Conclusion:', 'conclusion:']:
        parts = text_n.split(pat)
        if len(parts) > 1:
            for i in range(len(parts)-1, 0, -1):
                cand = parts[i].strip()
                for fp in ["### Reason", "Let's think step by step again", "let's go back and check", "###"]:
                    if fp in cand: cand = cand.split(fp)[0]
                if re.search(r'\(\d+\)', cand[:200]):
                    return cand.strip(), True
                if best is None: best = cand.strip()
    if best: return best, True
    pred, ok = parse_answer(pred_str=text_n)
    if ok and pred and pred.strip(): return pred, ok
    pred, ok = parse_answer(pred_str=text)
    if ok and pred and pred.strip(): return pred, ok
    return parse_answer(pred_str=raw_output)

def strip_markdown(t):
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t); t = t.replace("**", "")
    t = re.sub(r'\*(.*?)\*', r'\1', t); t = t.replace("`", "")
    return re.sub(r'\\text\{(.*?)\}', r'\1', t)

def normalize_cond(t):
    t = strip_markdown(t).strip().rstrip(".").lower()
    m = re.match(r'^(.+?)\s*[\u2013\u2014:\-]+\s*(knight|knave)s?$', t, re.IGNORECASE)
    if m: return f"{m.group(1).strip()} is a {m.group(2).strip().lower()}"
    m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)s?\.?$', t, re.IGNORECASE)
    if m: return f"{m.group(1).strip()} is a {m.group(2).strip().lower()}"
    return t

def improved_judge(pred, gold_conds):
    if not pred or not gold_conds: return False, "empty", 0.0
    cp = strip_markdown(pred)
    for fp in ["### Reason", "Let's think step by step again", "let's go back and check", "###"]:
        if fp in cp: cp = cp.split(fp)[0]
    cc = sum(1 for gc in gold_conds if gc.lower() in cp.lower())
    if cc == len(gold_conds): return True, "", 1.0
    pl = cp.strip().split("\n")
    pn = set()
    for line in pl:
        line = re.sub(r'^\(?\d+\)?[\.\):\s]+', '', line.strip()).strip()
        n = normalize_cond(line)
        if n: pn.add(n)
    gn = set(normalize_cond(g) for g in gold_conds if normalize_cond(g))
    if gn and gn.issubset(pn): return True, "", 1.0
    try:
        pd = parse_single_answer(pred)
        if pd:
            pc = set(f"{n.lower().strip()} is a {r.lower().strip()}" for n,r in pd.items())
            if gn and gn.issubset(pc): return True, "", 1.0
    except Exception: pass
    ccn = 0
    for g in gn:
        if g in cp.lower(): ccn += 1; continue
        m = re.match(r'^(.+?)\s+is\s+a\s+(knight|knave)', g)
        if m:
            ne = re.escape(m.group(1))
            if re.search(rf'{ne}.*?{m.group(2)}', cp.lower()): ccn += 1
    tot = len(gn) if gn else len(gold_conds)
    ratio = ccn/tot if tot > 0 else 0.0
    if ccn == tot: return True, "", ratio
    return False, "wrong_identity", ratio

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat()
    print(f"{MODEL} KandK Re-evaluation\nSource: {SOURCE_FILE}\n")
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        original = json.load(f)
    result = copy.deepcopy(original)
    total = len(result["records"])
    orig_perf = original["performance"]
    score0=0; rec_count=0; still=0; empty=0; changes=[]
    for rec in result["records"]:
        idx = rec["index"]; old = rec.get("score", 0)
        gt = rec.get("ground_truth", [])
        raw = rec.get("raw_output", "")
        if old == 1.0: continue
        score0 += 1
        if not raw or raw.strip() == "":
            empty += 1; continue
        pa, _ = parse_answer_improved(raw)
        if not pa or pa.strip() == "":
            empty += 1; continue
        gl = [gt] if isinstance(gt, str) else list(gt)
        ok, reason, ratio = improved_judge(pa, gl)
        rec["reeval_old_score"] = old
        rec["prediction"] = pa
        if ok:
            rec["score"] = 1.0; rec["reeval_recovered"] = True; rec_count += 1
            changes.append(f"  idx={idx}: RECOVERED 0->1")
        else:
            rec["score"] = 0.0; rec["reeval_recovered"] = False; still += 1
    new_pass = sum(1 for r in result["records"] if r.get("score", 0) == 1.0)
    new_perf = new_pass/total
    result["performance"] = new_perf
    result["_reeval_meta"] = {
        "script": "reeval_kandk_4bit.py", "timestamp": ts,
        "original_file": str(SOURCE_FILE),
        "original_performance": orig_perf, "corrected_performance": new_perf,
        "score0_count": score0, "recovered": rec_count, "still_wrong": still, "empty_prediction": empty}
    out_path = OUTPUT_DIR / f"{SOURCE_FILE.stem}-reeval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nTotal:{total}  Score=0:{score0}  Recovered:{rec_count}  Still:{still}  Empty:{empty}")
    print(f"Perf: {orig_perf*100:.2f}% -> {new_perf*100:.2f}% (+{(new_perf-orig_perf)*100:.2f}%)")
    print(f"Output: {out_path}")
    log_path = OUTPUT_DIR / "reeval_kandk_4bit_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"KandK 4bit reeval\n{ts}\nPerf: {orig_perf*100:.2f}% -> {new_perf*100:.2f}%\n\n" + "\n".join(changes))

if __name__ == "__main__":
    main()
