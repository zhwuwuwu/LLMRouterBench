"""
Single-query test script for SuperRouter API.
Usage: python test_single_query.py [index]
  - index: record index from checkpoint (default: 0)
  - Or use --dirty to test all 15 dirty records
"""
import sys
import json
import time
import os

# Ensure .env is loaded
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv(".env")

import httpx
from openai import OpenAI

API_KEY = os.environ.get("INTEL_SUPERROUTER_API_KEY", "sk-0IktBaUNDpzB6UVZeJW_Bg")
BASE_URL = "https://superrouter.intel.com/v1"
MODEL = "qwen3-coder-next-80b-4bit-awq"

CHECKPOINT_PATH = "results/bench/livecodebench/test/qwen3-coder-next/livecodebench-test-qwen3-coder-next-checkpoint.json"
DIRTY_INDICES = [452, 527, 529, 635, 666, 668, 669, 670, 671, 672, 674, 772, 798, 799, 800]


def create_client():
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=httpx.Client(verify=False),
    )


def test_query(client, prompt, index, timeout=600):
    print(f"\n{'='*60}")
    print(f"Testing record index={index}, prompt length={len(prompt)} chars")
    print(f"First 200 chars: {prompt[:200]}...")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=1.0,
            timeout=timeout,
        )
        elapsed = time.time() - start
        output = response.choices[0].message.content
        usage = response.usage
        print(f"\n✅ SUCCESS in {elapsed:.1f}s")
        print(f"   prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}")
        print(f"   Output preview ({len(output)} chars): {output[:300]}...")
        return True
    except Exception as e:
        elapsed = time.time() - start
        cause = e.__cause__
        cause_info = f" <- {type(cause).__name__}: {cause}" if cause else ""
        print(f"\n❌ FAILED in {elapsed:.1f}s")
        print(f"   Error: {type(e).__name__}: {e}{cause_info}")
        return False


def load_prompts_from_checkpoint(indices):
    """Load prompts from checkpoint file"""
    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    prompts = {}
    for idx in indices:
        rec = data["records"][idx]
        if rec and rec.get("prompt"):
            prompts[idx] = rec["prompt"]
        else:
            print(f"  Warning: record {idx} is None in checkpoint, loading from dataset...")
    
    # For None records, load from dataset
    missing = [idx for idx in indices if idx not in prompts]
    if missing:
        from evaluation.LiveCodeBench.livecodebench import LiveCodeBenchEvaluator
        evaluator = LiveCodeBenchEvaluator()
        dataset = evaluator.load_data("test")
        for idx in missing:
            prompts[idx] = dataset[idx]["prompt"]
    
    return prompts


def main():
    args = sys.argv[1:]
    
    if "--dirty" in args:
        indices = DIRTY_INDICES
        print(f"Testing all {len(indices)} dirty records...")
    elif args and args[0].isdigit():
        indices = [int(args[0])]
    else:
        indices = [0]  # default: first record
    
    # Load prompts
    prompts = load_prompts_from_checkpoint(indices)
    
    # Create client
    client = create_client()
    
    # Test connectivity first
    print("Testing API connectivity...")
    try:
        models = client.models.list()
        print(f"✅ Connected. Available models: {len(models.data)}")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print(f"   NO_PROXY={os.environ.get('NO_PROXY', '<not set>')}")
        print(f"   HTTPS_PROXY={os.environ.get('HTTPS_PROXY', '<not set>')}")
        return
    
    # Run tests
    results = {}
    for idx in indices:
        success = test_query(client, prompts[idx], idx)
        results[idx] = success
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {sum(results.values())}/{len(results)} succeeded")
    for idx, ok in results.items():
        print(f"  [{idx}] {'✅' if ok else '❌'}")


if __name__ == "__main__":
    main()
