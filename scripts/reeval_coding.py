"""
Re-evaluate coding benchmark results (HumanEval, MBPP, LiveCodeBench).

Fixes false negatives caused by multiprocessing.Process timeout on Windows:
- Original evaluation: concurrency=8, time_out=3s, multiple Process spawns competing
- Re-eval: sequential execution, time_out=15s, one Process at a time

Usage:
    python -m scripts.reeval_coding --dataset humaneval --model MiniMax-M2.7
    python -m scripts.reeval_coding --dataset mbpp --model MiniMax-M2.7
    python -m scripts.reeval_coding --dataset livecodebench --model MiniMax-M2.7
    python -m scripts.reeval_coding --dataset all --model MiniMax-M2.7
    python -m scripts.reeval_coding --dataset humaneval --model MiniMax-M2.7 --timeout 30
    python -m scripts.reeval_coding --dataset humaneval --model MiniMax-M2.7 --only-failures
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_result_file(dataset: str, model: str) -> Path:
    """Find the result JSON file for a given dataset and model."""
    base = PROJECT_ROOT / "results" / "bench" / dataset / "test" / model
    if not base.exists():
        raise FileNotFoundError(f"Result directory not found: {base}")
    
    # Find result JSON files (exclude checkpoint and doublecheck files)
    candidates = [
        f for f in base.glob(f"{dataset}-test-{model}-*.json")
        if "checkpoint" not in f.name and "doublecheck" not in f.name
    ]
    
    if not candidates:
        raise FileNotFoundError(f"No result file found in {base}")
    
    # Use the most recent one
    candidates.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return candidates[0]


def reeval_humaneval(record: dict, timeout: float) -> dict:
    """Re-evaluate a single HumanEval record."""
    from evaluation.HumanEval.execution import check_correctness
    
    prediction = record["prediction"]
    if not prediction or not prediction.strip():
        return {"passed": False, "detail": "empty prediction"}
    
    result = check_correctness(
        task_id=record["index"],
        completion_id=0,
        solution=prediction,
        time_out=timeout
    )
    return {"passed": result["passed"], "detail": result["result"]}


def reeval_mbpp(record: dict, timeout: float) -> dict:
    """Re-evaluate a single MBPP record."""
    from evaluation.MBPP.execution import check_correctness
    
    prediction = record["prediction"]
    if not prediction or not prediction.strip():
        return {"passed": False, "detail": "empty prediction"}
    
    result = check_correctness(
        task_id=record["index"],
        completion_id=0,
        solution=prediction,
        time_out=timeout
    )
    return {"passed": result["passed"], "detail": result["result"]}


def reeval_livecodebench(record: dict, timeout: float) -> dict:
    """Re-evaluate a single LiveCodeBench record.
    
    LiveCodeBench stores input/output in extra_fields or needs reconstruction
    from the original dataset. We re-extract from the raw data.
    """
    import json as json_mod
    import base64
    import zlib
    import pickle
    from evaluation.LiveCodeBench.compute_code_generation_metrics import evaluate_generation
    from evaluation.LiveCodeBench.livecodebench import LiveCodeBenchEvaluator
    
    prediction = record["prediction"]
    if not prediction or not prediction.strip():
        return {"passed": False, "detail": "empty prediction"}
    
    # We need the sample data (input_output) from the original dataset
    # This is stored in _reeval_sample injected by the caller
    sample = record.get("_reeval_sample")
    if sample is None:
        return {"passed": False, "detail": "missing sample data for re-eval"}
    
    results, metadata = evaluate_generation(
        generations=[prediction], sample=sample, debug=False, timeout=int(timeout)
    )
    real_results = results[0]
    
    correct_number = sum(1 for r in real_results if r is True)
    is_correct = correct_number == len(real_results)
    
    detail = "passed" if is_correct else f"failed: {correct_number}/{len(real_results)} tests passed"
    return {"passed": is_correct, "detail": detail}


def load_livecodebench_samples() -> dict:
    """Load LiveCodeBench dataset and build index→sample mapping.
    
    Returns dict mapping origin_query (question_content) to sample dict.
    """
    import json as json_mod
    import base64
    import zlib
    import pickle
    
    data_dir = PROJECT_ROOT / "data" / "LiveCodeBench"
    all_data = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    
    # Build mapping from question_content to sample
    samples = {}
    for raw_item in all_data:
        question_content = raw_item.get("question_content", "")
        
        # Decode test cases
        public_test_cases = json.loads(raw_item["public_test_cases"])
        try:
            private_test_cases = json.loads(raw_item["private_test_cases"])
        except Exception:
            try:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(raw_item["private_test_cases"].encode("utf-8"))
                        )
                    )
                )
            except Exception:
                private_test_cases = []
        
        fn_name = json.loads(raw_item["metadata"]).get("func_name", None)
        all_tests = public_test_cases + private_test_cases
        inputs = [t["input"] for t in all_tests]
        outputs = [t["output"] for t in all_tests]
        sample = {"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": fn_name})}
        
        samples[question_content] = sample
    
    return samples


def run_reeval(dataset: str, model: str, timeout: float, only_failures: bool):
    """Run re-evaluation for a dataset."""
    print(f"\n{'='*60}")
    print(f"Re-evaluating {dataset} for {model}")
    print(f"Timeout: {timeout}s | Only failures: {only_failures}")
    print(f"{'='*60}\n")
    
    result_file = find_result_file(dataset, model)
    print(f"Input file: {result_file}")
    
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    records = data["records"]
    total = len(records)
    print(f"Total records: {total}")
    
    # Select evaluator function
    lcb_samples: dict = {}
    if dataset == "humaneval":
        eval_fn = reeval_humaneval
    elif dataset == "mbpp":
        eval_fn = reeval_mbpp
    elif dataset == "livecodebench":
        eval_fn = reeval_livecodebench
        # Pre-load LiveCodeBench samples
        print("Loading LiveCodeBench dataset samples...")
        lcb_samples = load_livecodebench_samples()
        print(f"Loaded {len(lcb_samples)} samples")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Track changes
    changes_0_to_1 = []  # false negatives recovered
    changes_1_to_0 = []  # regressions (should be 0)
    errors = []
    skipped = 0
    
    for i, record in enumerate(records):
        old_score = record.get("score", 0.0)
        
        # Skip if only_failures and already passed
        if only_failures and old_score == 1.0:
            record["reeval_old_score"] = old_score
            record["reeval_detail"] = "skipped (already passed)"
            skipped += 1
            continue
        
        # Inject sample for LiveCodeBench
        if dataset == "livecodebench":
            origin_query = record.get("origin_query", "")
            sample = lcb_samples.get(origin_query)
            if sample is None:
                record["reeval_old_score"] = old_score
                record["reeval_detail"] = "error: sample not found"
                errors.append(record["index"])
                print(f"  [{i+1}/{total}] idx={record['index']} - ERROR: sample not found")
                continue
            record["_reeval_sample"] = sample
        
        # Run evaluation
        try:
            result = eval_fn(record, timeout)
            new_score = 1.0 if result["passed"] else 0.0
            detail = result["detail"]
        except Exception as e:
            new_score = 0.0
            detail = f"error: {str(e)}"
            errors.append(record["index"])
        
        # Clean up temporary fields
        if "_reeval_sample" in record:
            del record["_reeval_sample"]
        
        # Update record
        record["reeval_old_score"] = old_score
        record["reeval_detail"] = detail
        record["score"] = new_score
        
        # Track changes
        if old_score == 0.0 and new_score == 1.0:
            changes_0_to_1.append(record["index"])
        elif old_score == 1.0 and new_score == 0.0:
            changes_1_to_0.append(record["index"])
        
        # Progress
        status = ""
        if old_score != new_score:
            status = f" *** {old_score} -> {new_score} ***"
        
        if (i + 1) % 50 == 0 or old_score != new_score:
            print(f"  [{i+1}/{total}] idx={record['index']} old={old_score} new={new_score} detail={detail}{status}")
    
    # Recalculate performance
    scored_records = [r for r in records if r.get("score") is not None]
    new_performance = sum(r["score"] for r in scored_records) / len(scored_records) if scored_records else 0.0
    old_performance = data["performance"]
    data["performance"] = new_performance
    
    # Save doublecheck file
    output_file = result_file.with_name(result_file.stem + "_doublecheck.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset} / {model}")
    print(f"{'='*60}")
    print(f"Original performance: {old_performance:.4f} ({int(old_performance * total)}/{total})")
    print(f"Re-eval performance:  {new_performance:.4f} ({int(new_performance * len(scored_records))}/{len(scored_records)})")
    print(f"")
    print(f"False negatives recovered (0→1): {len(changes_0_to_1)}")
    if changes_0_to_1:
        print(f"  Indices: {changes_0_to_1}")
    print(f"Regressions (1→0):               {len(changes_1_to_0)}")
    if changes_1_to_0:
        print(f"  Indices: {changes_1_to_0}")
        print(f"  *** WARNING: Regressions detected! This should not happen. ***")
    print(f"Errors:                           {len(errors)}")
    if errors:
        print(f"  Indices: {errors}")
    print(f"Skipped (already passed):         {skipped}")
    print(f"")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}\n")
    
    return {
        "dataset": dataset,
        "model": model,
        "old_performance": old_performance,
        "new_performance": new_performance,
        "changes_0_to_1": len(changes_0_to_1),
        "changes_1_to_0": len(changes_1_to_0),
        "errors": len(errors),
        "output_file": str(output_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate coding benchmark results")
    parser.add_argument("--dataset", required=True, choices=["humaneval", "mbpp", "livecodebench", "all"],
                        help="Dataset to re-evaluate")
    parser.add_argument("--model", required=True, help="Model name (e.g., MiniMax-M2.7)")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Timeout in seconds for each evaluation (default: 15)")
    parser.add_argument("--only-failures", action="store_true",
                        help="Only re-evaluate records with score=0 (faster)")
    args = parser.parse_args()
    
    datasets = ["humaneval", "mbpp", "livecodebench"] if args.dataset == "all" else [args.dataset]
    
    results = []
    for ds in datasets:
        try:
            result = run_reeval(ds, args.model, args.timeout, args.only_failures)
            results.append(result)
        except FileNotFoundError as e:
            print(f"Skipping {ds}: {e}")
        except Exception as e:
            print(f"Error processing {ds}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY OF ALL RE-EVALUATIONS")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['dataset']:15s}: {r['old_performance']:.4f} -> {r['new_performance']:.4f} "
                  f"(+{r['changes_0_to_1']} recovered, {r['changes_1_to_0']} regressions)")


if __name__ == "__main__":
    main()
