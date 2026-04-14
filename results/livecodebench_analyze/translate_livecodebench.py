"""
Enrich LiveCodeBench comparison CSV with full problem text + Chinese translation.
Sequential sync version using httpx with trust_env=False to bypass corporate proxy.
"""
import csv
import json
import time
from pathlib import Path

import httpx

# --- Config ---
ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "http://localhost:4000/v1"
API_KEY = "sk-GHJGDcR8lurojxdlmHEIzw"
MODEL = "gemini-3.1-pro-preview"

QWEN_RESULT = ROOT / "results" / "bench" / "livecodebench" / "test" / "qwen3-coder-next" / "livecodebench-test-qwen3-coder-next-20260325_090311.json"
GPT5_RESULT = ROOT / "results" / "bench" / "livecodebench" / "test" / "gpt-5" / "livecodebench-test-gpt-5-20251013_115531.json"
OUTPUT_CSV = ROOT / "results" / "minimax_reeval" / "livecodebench_qwen_vs_gpt5.csv"
CHECKPOINT = ROOT / "results" / "minimax_reeval" / "_translate_checkpoint.json"

TIMEOUT = 120
CHECKPOINT_INTERVAL = 20


def translate_one(client: httpx.Client, text: str) -> str:
    """Translate a single problem to Chinese. Retries up to 3 times."""
    prompt = (
        "你是一个专业翻译。请将以下编程竞赛题目翻译成中文。"
        "只输出翻译结果，不要加任何解释或前缀。\n\n"
        f"{text}"
    )
    for attempt in range(3):
        try:
            resp = client.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                return f"[TRANSLATION_ERROR: {e}]"
    return "[TRANSLATION_ERROR: unreachable]"


def main():
    # Load results
    with open(QWEN_RESULT, "r", encoding="utf-8") as f:
        qwen = json.load(f)
    with open(GPT5_RESULT, "r", encoding="utf-8") as f:
        gpt5 = json.load(f)

    records = []
    for i in range(len(qwen["records"])):
        qr = qwen["records"][i]
        gr = gpt5["records"][i]
        q_pass = qr["score"] == 1.0
        g_pass = gr["score"] == 1.0

        if q_pass and g_pass:
            comp = "both_pass"
        elif not q_pass and not g_pass:
            comp = "both_fail"
        elif q_pass and not g_pass:
            comp = "qwen_only"
        else:
            comp = "gpt5_only"

        records.append({
            "index": qr["index"],
            "origin_query": qr["origin_query"],
            "qwen3_coder_next_pass": int(q_pass),
            "gpt5_pass": int(g_pass),
            "comparison": comp,
            "chinese_translation": "",
        })

    # Load checkpoint if exists, filtering out failed translations
    translated = {}
    if CHECKPOINT.exists():
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            raw = json.load(f)
        translated = {k: v for k, v in raw.items() if not v.startswith("[TRANSLATION_ERROR")}
        skipped = len(raw) - len(translated)
        print(f"Loaded {len(translated)} cached translations from checkpoint (skipped {skipped} errors).")

    # Find what still needs translating
    to_translate = []
    for rec in records:
        idx_str = str(rec["index"])
        if idx_str in translated:
            rec["chinese_translation"] = translated[idx_str]
        else:
            to_translate.append(rec)

    total = len(to_translate)
    print(f"Total: {len(records)}, Already translated: {len(translated)}, Remaining: {total}")

    if to_translate:
        # trust_env=False: bypass Windows system proxy (Intel corporate proxy blocks localhost)
        with httpx.Client(trust_env=False) as client:
            for i, rec in enumerate(to_translate):
                idx = rec["index"]
                translation = translate_one(client, rec["origin_query"])
                translated[str(idx)] = translation
                rec["chinese_translation"] = translation

                done = i + 1
                if done % 10 == 0 or done == total:
                    print(f"  [{done}/{total}] translated")

                if done % CHECKPOINT_INTERVAL == 0:
                    with open(CHECKPOINT, "w", encoding="utf-8") as f:
                        json.dump(translated, f, ensure_ascii=False)

            # Final checkpoint
            with open(CHECKPOINT, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False)

    # Apply cached translations to records that were skipped
    for rec in records:
        idx_str = str(rec["index"])
        if idx_str in translated and not rec["chinese_translation"]:
            rec["chinese_translation"] = translated[idx_str]

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csvf:
        writer = csv.writer(csvf)
        writer.writerow([
            "index", "problem_full", "problem_chinese",
            "qwen3_coder_next_pass", "gpt5_pass", "comparison",
        ])
        for rec in records:
            problem_full = rec["origin_query"].replace("\r\n", "\\n").replace("\n", "\\n")
            problem_chinese = rec["chinese_translation"].replace("\r\n", "\\n").replace("\n", "\\n")
            writer.writerow([
                rec["index"], problem_full, problem_chinese,
                rec["qwen3_coder_next_pass"], rec["gpt5_pass"], rec["comparison"],
            ])

    print(f"\nCSV saved to: {OUTPUT_CSV}")
    print(f"Total rows: {len(records)}")

    both_pass = sum(1 for r in records if r["comparison"] == "both_pass")
    both_fail = sum(1 for r in records if r["comparison"] == "both_fail")
    qwen_only = sum(1 for r in records if r["comparison"] == "qwen_only")
    gpt5_only = sum(1 for r in records if r["comparison"] == "gpt5_only")
    errors = sum(1 for v in translated.values() if v.startswith("[TRANSLATION_ERROR"))
    print(f"Both pass: {both_pass}, Both fail: {both_fail}, Qwen only: {qwen_only}, GPT-5 only: {gpt5_only}")
    print(f"Translation errors: {errors}")


if __name__ == "__main__":
    main()
