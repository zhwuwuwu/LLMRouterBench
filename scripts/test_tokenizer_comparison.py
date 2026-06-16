"""
Tokenizer comparison test: Send identical prompts to 3 models, compare token counts.

Test design:
  - Prompt: Ask each model to recite the first 4 lines of 静夜思 (Li Bai)
  - Expected output: All models produce nearly identical text
  - Compare: prompt_tokens and completion_tokens across models
  - This isolates tokenizer differences from content differences

Usage: python scripts/test_tokenizer_comparison.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(".env", override=True)

import httpx
from openai import OpenAI
import os
# Set NO_PROXY environment variable for internal Intel services
os.environ.setdefault("NO_PROXY", "superrouter.intel.com")

# ── Model configurations ──────────────────────────────────────────
MODELS = [
    {
        "name": "MiniMax-M2.7",
        "api_model_name": "MiniMax-M2.7",
        "base_url": "https://api.minimaxi.com/v1",
        "api_key": "sk-api-8rS3Tj0l--tEoAFlHn6m5mqYmlGZXaas1mQ_rA_zXNvf4lXmI6Mt-2-DRR7JQj6nJ0iJROxUXWgxM2AeSdML_H-OhVmXKcSMZU-VwoRT1qztJ2OwvQvfshE",
        "use_proxy": True,
    },
    {
        "name": "qwen3-coder-next",
        "api_model_name": "qwen3-coder-next-80b-4bit-awq",
        "base_url": "https://superrouter.intel.com/v1",
        "api_key": "sk-QR8xBakQQHV-CVhPHH4E2g",
        "use_proxy": False,
    },
    # GPT-5 (aihub) removed — service is down
]

# ── Test prompts ──────────────────────────────────────────────────
# Use a very controlled prompt so output should be (nearly) identical
PROMPT_CN = '请你只输出李白《静夜思》的前四句，不要输出任何其他内容，不要解释，不要加标点以外的任何文字。直接输出诗句：'
PROMPT_EN = 'Output only the first 4 lines of the poem "Twinkle Twinkle Little Star" by Jane Taylor. Do not add any explanation or extra text. Output the lines directly:'

PROMPTS = {
    "中文-静夜思": PROMPT_CN,
    "英文-Twinkle Star": PROMPT_EN,
}


def create_client(model_cfg):
    """Create OpenAI client with appropriate proxy settings."""
    if model_cfg["use_proxy"]:
        proxy_url = os.environ.get("HTTPS_PROXY", "http://proxy-iil.intel.com:912")
        http_client = httpx.Client(proxy=proxy_url, verify=False, timeout=300)
    else:
        # For internal Intel services: NO_PROXY is set via .env to bypass proxy.
        http_client = httpx.Client(verify=False, timeout=300)

    return OpenAI(
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        http_client=http_client,
    )


def test_model(client, model_cfg, prompt, temperature=0.0):
    """Send prompt to model and return usage info."""
    try:
        response = client.chat.completions.create(
            model=model_cfg["api_model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=1.0,
            timeout=300,
        )
        output = response.choices[0].message.content
        usage = response.usage
        return {
            "success": True,
            "output": output,
            "output_chars": len(output),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    print("=" * 80)
    print("  Tokenizer Comparison Test: Identical Prompt → Compare Token Counts")
    print("=" * 80)

    results = {}

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n{'─' * 80}")
        print(f"  Prompt: {prompt_name}")
        print(f"  Text ({len(prompt_text)} chars): {prompt_text}")
        print(f"{'─' * 80}")

        results[prompt_name] = {}

        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            print(f"\n  [{model_name}] Calling API...", end=" ", flush=True)

            client = create_client(model_cfg)
            start = time.time()
            result = test_model(client, model_cfg, prompt_text)
            elapsed = time.time() - start

            if result["success"]:
                print(f"OK ({elapsed:.1f}s)")
                print(f"    prompt_tokens:     {result['prompt_tokens']}")
                print(f"    completion_tokens: {result['completion_tokens']}")
                print(f"    total_tokens:      {result['total_tokens']}")
                print(f"    output ({result['output_chars']} chars):  {result['output'][:200]}")
                result["time"] = round(elapsed, 2)
            else:
                print(f"FAILED ({elapsed:.1f}s)")
                print(f"    error: {result['error']}")

            results[prompt_name][model_name] = result

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  SUMMARY: Token Count Comparison")
    print(f"{'=' * 80}")

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n  Prompt: {prompt_name} ({len(prompt_text)} chars)")
        print(f"  {'Model':<20} {'PT':>8} {'CT':>8} {'Total':>8} {'Output chars':>12} {'chars/PT':>10}")
        print(f"  {'─' * 66}")

        prompt_results = results[prompt_name]
        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            r = prompt_results.get(model_name, {})
            if r.get("success"):
                chars_per_pt = len(prompt_text) / r["prompt_tokens"] if r["prompt_tokens"] > 0 else 0
                print(f"  {model_name:<20} {r['prompt_tokens']:>8} {r['completion_tokens']:>8} "
                      f"{r['total_tokens']:>8} {r['output_chars']:>12} {chars_per_pt:>10.2f}")
            else:
                print(f"  {model_name:<20} {'FAILED':>8}")

    # ── Analysis ──────────────────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("  ANALYSIS")
    print(f"{'=' * 80}")

    for prompt_name in PROMPTS:
        prompt_results = results[prompt_name]
        successful = {k: v for k, v in prompt_results.items() if v.get("success")}
        if len(successful) < 2:
            continue

        pts = {k: v["prompt_tokens"] for k, v in successful.items()}
        cts = {k: v["completion_tokens"] for k, v in successful.items()}

        min_pt_model = min(pts, key=pts.get)
        max_pt_model = max(pts, key=pts.get)
        min_ct_model = min(cts, key=cts.get)
        max_ct_model = max(cts, key=cts.get)

        print(f"\n  [{prompt_name}]")
        print(f"  Prompt tokens range: {pts[min_pt_model]} ({min_pt_model}) ~ {pts[max_pt_model]} ({max_pt_model})")
        if pts[min_pt_model] > 0:
            pt_diff_pct = (pts[max_pt_model] - pts[min_pt_model]) / pts[min_pt_model] * 100
            print(f"  PT difference: {pt_diff_pct:.1f}% (max vs min)")

        print(f"  Completion tokens range: {cts[min_ct_model]} ({min_ct_model}) ~ {cts[max_ct_model]} ({max_ct_model})")

        # Check if outputs are identical
        outputs = {k: v["output"].strip() for k, v in successful.items()}
        unique_outputs = set(outputs.values())
        if len(unique_outputs) == 1:
            print(f"  Output: IDENTICAL across all models ✓")
            print(f"  → CT difference is purely tokenizer efficiency (same text, different token counts)")
        else:
            print(f"  Output: DIFFERS ({len(unique_outputs)} unique outputs)")
            for name, text in outputs.items():
                print(f"    {name}: {text[:100]}")

    # ── Save results ──────────────────────────────────────────────
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results", "minimax_reeval", "tokenizer_comparison_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
