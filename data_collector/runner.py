import re
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path
from loguru import logger

from .config_loader import BenchmarkConfig
from .planner import RunPlan
from .storage import ResultsStorage, BenchmarkResult, RecordResult
from evaluation.factory import EvaluatorFactory
from generators.factory import create_generator

# Checkpoint constants
CHECKPOINT_INTERVAL_RECORDS = 50   # Save every N completed records
CHECKPOINT_INTERVAL_SECONDS = 300  # Save every N seconds (5 min)


def extract_field_from_response(response_json: str, path: str) -> Any:
    """
    Extract a field from response JSON string by path.

    Supported path formats:
    - "usage.prompt_tokens" -> response["usage"]["prompt_tokens"]
    - "choices[0].message.content" -> response["choices"][0]["message"]["content"]

    Args:
        response_json: API response JSON string
        path: Field path string

    Returns:
        Extracted value, or None if extraction fails
    """
    if not response_json or not path:
        return None

    try:
        response = json.loads(response_json)
    except json.JSONDecodeError:
        return None

    current = response
    # Split path: "choices[0].message.content" -> ["choices", "0", "message", "content"]
    tokens = re.split(r'\.|\[|\]', path)
    tokens = [t for t in tokens if t]  # Remove empty strings

    try:
        for token in tokens:
            if token.isdigit():
                # Index access
                current = current[int(token)]
            elif isinstance(current, dict):
                current = current[token]
            else:
                return None
        return current
    except (KeyError, IndexError, TypeError):
        return None


def extract_extra_fields(response_json: str, extract_config: Dict[str, str]) -> Dict[str, Any]:
    """
    Extract multiple fields from response based on config.

    Args:
        response_json: API response JSON string
        extract_config: {field_name: path} dictionary

    Returns:
        {field_name: value} dictionary, only contains successfully extracted fields
    """
    if not response_json or not extract_config:
        return {}

    result = {}
    for field_name, path in extract_config.items():
        value = extract_field_from_response(response_json, path)
        if value is not None:
            result[field_name] = value
    return result


class BenchmarkRunner:
    """Execute benchmark runs with concurrent processing"""

    def __init__(self, config: BenchmarkConfig, storage: ResultsStorage):
        self.config = config
        self.storage = storage
        self.evaluator_factory = EvaluatorFactory(grader_cache_config=config.grader_cache_config)
    
    def run_all(self, plans: List[RunPlan]) -> Dict[str, Any]:
        """Execute all planned runs"""
        if not plans:
            logger.info("No runs to execute")
            return {"total_runs": 0, "successful_runs": 0, "failed_runs": 0}
        
        if self.config.run.demo_mode:
            logger.info(f"Starting execution in DEMO MODE - limiting each dataset to {self.config.run.demo_limit} records")
            logger.info(f"Demo mode: {len(plans)} runs planned (results will not be indexed)")
        else:
            logger.info(f"Starting execution of {len(plans)} runs")
        start_time = time.time()
        
        results = {
            "total_runs": len(plans),
            "successful_runs": 0,
            "failed_runs": 0,
            "run_details": []
        }
        
        for plan in plans:
            logger.info(f"Processing run: {plan.run_key}")
            
            try:
                run_result = self.execute_single_run(plan)
                if run_result:
                    results["successful_runs"] += 1
                    results["run_details"].append({
                        "run_key": plan.run_key,
                        "status": "success",
                        "performance": run_result.performance,
                        "time_taken": run_result.time_taken
                    })
                else:
                    results["failed_runs"] += 1
                    results["run_details"].append({
                        "run_key": plan.run_key,
                        "status": "failed",
                        "error": "Unknown error"
                    })
            except Exception as e:
                logger.error(f"Failed to execute run {plan.run_key}: {str(e)}")
                results["failed_runs"] += 1
                results["run_details"].append({
                    "run_key": plan.run_key,
                    "status": "failed",
                    "error": str(e)
                })
        
        total_time = time.time() - start_time
        logger.info(f"Completed {results['successful_runs']}/{results['total_runs']} runs in {total_time:.2f}s")
        
        return results
    
    def execute_single_run(self, plan: RunPlan) -> BenchmarkResult:
        """Execute a single benchmark run"""
        start_time = time.time()
        
        try:
            # 1. Get evaluator for this dataset
            evaluator = self.evaluator_factory.get_evaluator(plan.dataset_id)
            
            # 2. Load data (format_prompt already done in load_data)
            data = evaluator.load_data(plan.split)

            if not data:
                raise ValueError(f"No data loaded for {plan.dataset_id}/{plan.split}")

            # Calculate data fingerprint (before any modifications)
            data_fingerprint = self.storage.calculate_data_fingerprint(data)

            # Demo mode: limit data size
            if self.config.run.demo_mode:
                original_size = len(data)
                # Handle different data types properly
                if hasattr(data, 'select'):  # Dataset object
                    data = data.select(range(min(self.config.run.demo_limit, len(data))))
                else:  # List or other sequence
                    data = data[:self.config.run.demo_limit]
                logger.info(f"Demo mode: limited to {len(data)} records (from {original_size} total)")
            else:
                logger.info(f"Loaded {len(data)} records for {plan.dataset_id}/{plan.split}")
            
            # 3. Get model configuration
            model_config = self._get_model_config(plan.model_name)
            if not model_config:
                raise ValueError(f"Model configuration not found: {plan.model_name}")
            
            # 4. Create generator
            generator = create_generator(
                model_config=model_config.__dict__,
                cache_config=self.config.cache_config
            )
            logger.info(f"Created generator: {type(generator).__name__} for model {plan.model_name}")
            
            # 5. Process records concurrently
            records = self._process_records_concurrent(
                data=data,
                generator=generator,
                evaluator=evaluator,
                concurrency=self.config.run.concurrency,
                model_config=model_config,
                dataset_id=plan.dataset_id,
                split=plan.split,
                model_name=plan.model_name
            )
            
            # 6. Calculate aggregated statistics
            performance, total_prompt_tokens, total_completion_tokens, total_cost = self._calculate_aggregates(records)

            # 7. Calculate extra metrics (aggregated from record extra_fields)
            extra_metrics = self._calculate_extra_metrics(records, plan.dataset_id)

            # 8. Create result object
            result = BenchmarkResult(
                performance=performance,
                time_taken=time.time() - start_time,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                cost=total_cost,
                counts=len(records),
                model_name=plan.model_name,
                dataset_name=plan.dataset_id,
                split=plan.split,
                demo=self.config.run.demo_mode,
                records=records,
                extra_metrics=extra_metrics
            )

            # 9. Save result
            self.storage.save_result(result, plan.dataset_id, plan.split, plan.model_name, data_fingerprint)

            # 10. Clean up checkpoint file (no longer needed after successful save)
            self._delete_checkpoint(plan.dataset_id, plan.split, plan.model_name)

            logger.info(f"Completed run {plan.run_key}: {performance:.3f} performance")

            # 对 SGI-Bench 数据集输出额外指标汇总
            if extra_metrics:
                self._log_extra_metrics_summary(extra_metrics, plan.dataset_id)

            return result
            
        except Exception as e:
            logger.error(f"Error executing run {plan.run_key}: {str(e)}")
            raise
    
    # ─── Checkpoint helpers ───────────────────────────────────────────

    def _checkpoint_path(self, dataset_id: str, split: str, model_name: str) -> Path:
        """Return the canonical checkpoint file path."""
        return (self.storage.bench_dir / dataset_id / split / model_name /
                f"{dataset_id}-{split}-{model_name}-checkpoint.json")

    def _save_checkpoint(
        self,
        results: List[Optional[RecordResult]],
        dataset_id: str,
        split: str,
        model_name: str,
        total_count: int,
    ) -> None:
        """Atomically save a checkpoint of completed records so far."""
        cp_path = self._checkpoint_path(dataset_id, split, model_name)
        cp_path.parent.mkdir(parents=True, exist_ok=True)

        completed_count = sum(1 for r in results if r is not None)
        serialized_records = []
        for r in results:
            if r is None:
                serialized_records.append(None)
            else:
                serialized_records.append({
                    "index": r.index,
                    "origin_query": r.origin_query,
                    "prompt": r.prompt,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "cost": r.cost,
                    "score": r.score,
                    "prediction": r.prediction,
                    "ground_truth": r.ground_truth,
                    "raw_output": r.raw_output,
                    "extra_fields": r.extra_fields,
                })

        payload = {
            "checkpoint": True,
            "completed_count": completed_count,
            "total_count": total_count,
            "dataset_id": dataset_id,
            "split": split,
            "model_name": model_name,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "records": serialized_records,
        }

        tmp_path = cp_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(str(tmp_path), str(cp_path))
        logger.info(f"Checkpoint saved: {completed_count}/{total_count} records -> {cp_path.name}")

    def _load_checkpoint(
        self,
        dataset_id: str,
        split: str,
        model_name: str,
        total_count: int,
    ) -> Tuple[List[Optional[RecordResult]], Set[int]]:
        """
        Load a checkpoint if one exists and its total_count matches.

        Returns:
            (results_list, completed_indices) — results_list has RecordResult
            at completed positions and None elsewhere; completed_indices is
            the set of 0-based indices already done.
        """
        cp_path = self._checkpoint_path(dataset_id, split, model_name)
        if not cp_path.exists():
            return [None] * total_count, set()

        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if not payload.get("checkpoint"):
                logger.warning(f"Invalid checkpoint file (missing flag): {cp_path}")
                return [None] * total_count, set()

            stored_total = payload.get("total_count", 0)
            if stored_total != total_count:
                logger.warning(
                    f"Checkpoint total_count mismatch ({stored_total} vs {total_count}), ignoring checkpoint"
                )
                return [None] * total_count, set()

            raw_records = payload.get("records", [])
            if len(raw_records) != total_count:
                logger.warning(
                    f"Checkpoint records length mismatch ({len(raw_records)} vs {total_count}), ignoring"
                )
                return [None] * total_count, set()

            results: List[Optional[RecordResult]] = []
            completed: Set[int] = set()
            for idx, rec in enumerate(raw_records):
                if rec is None:
                    results.append(None)
                else:
                    results.append(RecordResult(**rec))
                    completed.add(idx)

            logger.info(
                f"Checkpoint loaded: {len(completed)}/{total_count} records from {cp_path.name}"
            )
            return results, completed

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {cp_path}: {e}")
            return [None] * total_count, set()

    def _delete_checkpoint(self, dataset_id: str, split: str, model_name: str) -> None:
        """Remove checkpoint file after successful save_result."""
        cp_path = self._checkpoint_path(dataset_id, split, model_name)
        if cp_path.exists():
            cp_path.unlink()
            logger.info(f"Checkpoint deleted: {cp_path.name}")

    # ─── Core processing ───────────────────────────────────────────

    def _process_records_concurrent(self, data: List[Dict[str, Any]], generator, evaluator, concurrency: int, model_config=None, dataset_id: str = "", split: str = "", model_name: str = "") -> List[RecordResult]:
        """Process records with concurrent execution and checkpoint support."""
        from tqdm import tqdm

        total_count = len(data)
        use_checkpoint = bool(dataset_id and split and model_name)

        # ── Load checkpoint (resume) ────────────────────────────────
        if use_checkpoint:
            results, completed_indices = self._load_checkpoint(
                dataset_id, split, model_name, total_count
            )
            skipped = len(completed_indices)
        else:
            results = [None] * total_count
            completed_indices: Set[int] = set()
            skipped = 0

        remaining_indices = [i for i in range(total_count) if i not in completed_indices]

        # Shuffle remaining indices to avoid clustering of hard/timeout-prone
        # records that may block all concurrent slots simultaneously.
        import random
        random.shuffle(remaining_indices)

        if skipped > 0:
            logger.info(
                f"Resuming from checkpoint: {skipped}/{total_count} already done, "
                f"{len(remaining_indices)} remaining"
            )

        if not remaining_indices:
            logger.info("All records already completed in checkpoint, nothing to do")
            return results  # type: ignore[return-value]

        def process_single_record(record_data: Dict[str, Any], index: int) -> RecordResult:
            """Process a single record"""
            thread_name = threading.current_thread().name
            start_ts = time.time()
            try:
                # Extract required fields
                origin_query = record_data.get('origin_query', record_data.get('question', ''))
                prompt = record_data.get('prompt', record_data.get('formatted_prompt', ''))
                
                if not prompt:
                    raise ValueError("No prompt found in record")
                
                logger.info(f"[{thread_name}][rec={index}] Starting record {index}, prompt_len={len(prompt)}")
                
                # Generate response (with images for multimodal generators)
                images = record_data.get('image_paths', [])
                if images:
                    gen_output = generator.generate(prompt, images=images)
                else:
                    gen_output = generator.generate(prompt)
                
                # Evaluate response using the correct interface
                raw_output = gen_output.output
                eval_result = evaluator.evaluate(record_data, raw_output)
                
                # Extract evaluation results
                # Handle boolean, decimal, and None is_correct values
                is_correct_value = eval_result.get('is_correct', False)
                if is_correct_value is None:
                    # Handle evaluators without internal scoring (e.g., ArenaHard)
                    score = None
                elif isinstance(is_correct_value, bool):
                    score = 1.0 if is_correct_value else 0.0
                else:
                    # Handle decimal values (0-1 range)
                    score = float(is_correct_value)
                prediction = eval_result.get('prediction', '')
                ground_truth = eval_result.get('ground_truth', '')

                # Extract extra fields from eval_result (除 is_correct/prediction/ground_truth 外的所有字段)
                extra_fields = {k: v for k, v in eval_result.items()
                                if k not in ('is_correct', 'prediction', 'ground_truth')}

                # 如果有 extract_fields 配置，再从 raw_response 提取并合并
                if model_config and model_config.extract_fields and gen_output.raw_response:
                    extra_fields.update(extract_extra_fields(
                        gen_output.raw_response,
                        model_config.extract_fields
                    ))

                elapsed = time.time() - start_ts
                logger.info(f"[{thread_name}][rec={index}] Completed record {index} in {elapsed:.1f}s, score={score}")
                return RecordResult(
                    index=index + 1,  # 1-based indexing
                    origin_query=origin_query,
                    prompt=prompt,
                    prompt_tokens=gen_output.prompt_tokens,
                    completion_tokens=gen_output.completion_tokens,
                    cost=gen_output.cost,
                    score=score,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    raw_output=raw_output,
                    extra_fields=extra_fields
                )
                
            except Exception as e:
                elapsed = time.time() - start_ts
                logger.warning(f"[{thread_name}][rec={index}] Failed record {index} in {elapsed:.1f}s: {str(e)}")
                return RecordResult(
                    index=index + 1,
                    origin_query=record_data.get('origin_query', record_data.get('question', '')),
                    prompt=record_data.get('prompt', record_data.get('formatted_prompt', '')),
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost=0.0,
                    score=None,
                    prediction="",
                    ground_truth="",
                    raw_output=f"Processing failed: {str(e)}"
                )

        # ── Checkpoint tracking state ───────────────────────────────
        completed_this_run = 0
        last_checkpoint_time = time.time()
        records_since_checkpoint = 0

        # ── Execute concurrently with progress tracking ─────────────
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # Only submit tasks for records NOT already in checkpoint
            future_to_index = {
                executor.submit(process_single_record, data[idx], idx): idx
                for idx in remaining_indices
            }
            
            # Collect results with progress bar (show total, start from skipped)
            with tqdm(total=total_count, initial=skipped, desc="Processing records", unit="record") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        logger.error(f"Unexpected error processing record {idx}: {e}")
                        results[idx] = RecordResult(
                            index=idx + 1,
                            origin_query="",
                            prompt="",
                            prompt_tokens=0,
                            completion_tokens=0,
                            cost=0.0,
                            score=None,
                            prediction="",
                            ground_truth="",
                            raw_output=f"Unexpected error: {str(e)}"
                        )
                    
                    pbar.update(1)
                    completed_this_run += 1
                    records_since_checkpoint += 1

                    # ── Periodic checkpoint save ────────────────────
                    if use_checkpoint:
                        now = time.time()
                        should_save = (
                            records_since_checkpoint >= CHECKPOINT_INTERVAL_RECORDS
                            or (now - last_checkpoint_time) >= CHECKPOINT_INTERVAL_SECONDS
                        )
                        if should_save:
                            self._save_checkpoint(
                                results, dataset_id, split, model_name, total_count
                            )
                            last_checkpoint_time = now
                            records_since_checkpoint = 0

        # ── Final checkpoint (in case last batch didn't trigger) ────
        if use_checkpoint and records_since_checkpoint > 0:
            self._save_checkpoint(results, dataset_id, split, model_name, total_count)

        return results  # type: ignore[return-value]
    
    def _calculate_aggregates(self, records: List[RecordResult]) -> tuple[float, int, int, float]:
        """Calculate aggregated statistics from records"""
        total_prompt_tokens = sum(r.prompt_tokens for r in records)
        total_completion_tokens = sum(r.completion_tokens for r in records)
        total_cost = sum(r.cost for r in records)
        
        # Calculate performance (average of non-null scores)
        valid_scores = [r.score for r in records if r.score is not None]
        performance = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        return performance, total_prompt_tokens, total_completion_tokens, total_cost
    
    def _get_model_config(self, model_name: str):
        """Get model configuration by name"""
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None

    def _calculate_extra_metrics(self, records: List[RecordResult], dataset_id: str) -> Dict[str, Any]:
        """
        Calculate aggregated extra metrics from record extra_fields.

        For SGI-Bench datasets, calculates specific metrics.
        For other datasets, aggregates all numeric fields.
        """
        if not records:
            return {}

        # 收集所有 extra_fields
        all_extra = [r.extra_fields for r in records if r.extra_fields]
        if not all_extra:
            return {}

        dataset_lower = dataset_id.lower()
        metrics = {}

        # SGI-Bench 特定指标
        if "sgibench" in dataset_lower:
            if "deepresearch" in dataset_lower:
                metrics["exact_match"] = sum(e.get("exact_match", 0) for e in all_extra) / len(all_extra)
                metrics["step_level_acc"] = sum(e.get("step_level_acc", 0) for e in all_extra) / len(all_extra)

            elif "dryexperiment" in dataset_lower:
                metrics["PassAll@5"] = sum(e.get("PassAll@5", 0) for e in all_extra) / len(all_extra)
                metrics["PassAll@3"] = sum(e.get("PassAll@3", 0) for e in all_extra) / len(all_extra)
                metrics["PassAll@1"] = sum(e.get("PassAll@1", 0) for e in all_extra) / len(all_extra)
                metrics["SER"] = sum(e.get("SER", 0) for e in all_extra) / len(all_extra)
                # AET 只统计有效值 (>0)
                valid_aet = [e.get("AET", -1) for e in all_extra if e.get("AET", -1) > 0]
                metrics["AET"] = sum(valid_aet) / len(valid_aet) if valid_aet else -1

            elif "wetexperiment" in dataset_lower:
                metrics["action_sequence_similarity"] = sum(e.get("action_sequence_similarity", 0) for e in all_extra) / len(all_extra)
                metrics["parameter_accuracy"] = sum(e.get("parameter_accuracy", 0) for e in all_extra) / len(all_extra)
                metrics["final_score"] = sum(e.get("final_score", 0) for e in all_extra) / len(all_extra)

            elif "ideageneration" in dataset_lower:
                metrics["final_score"] = sum(e.get("final_score", 0) for e in all_extra) / len(all_extra)
                metrics["effectiveness"] = sum(e.get("effectiveness", 0) for e in all_extra) / len(all_extra)
                metrics["novelty"] = sum(e.get("novelty", 0) for e in all_extra) / len(all_extra)
                metrics["detailedness"] = sum(e.get("detailedness", 0) for e in all_extra) / len(all_extra)
                metrics["feasibility"] = sum(e.get("feasibility", 0) for e in all_extra) / len(all_extra)

        else:
            # 通用逻辑：聚合所有数值型字段
            numeric_fields = {}
            for extra in all_extra:
                for key, value in extra.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)

            for key, values in numeric_fields.items():
                if values:
                    metrics[key] = sum(values) / len(values)

        return metrics

    def _log_extra_metrics_summary(self, extra_metrics: Dict[str, Any], dataset_id: str):
        """输出额外指标汇总日志"""
        if not extra_metrics:
            return

        dataset_lower = dataset_id.lower()

        # SGI-Bench 格式化输出
        if "sgibench" in dataset_lower:
            if "deepresearch" in dataset_lower:
                logger.info(f"  SGI-Bench metrics: exact_match={extra_metrics.get('exact_match', 0):.3f}, step_level_acc={extra_metrics.get('step_level_acc', 0):.3f}")
            elif "dryexperiment" in dataset_lower:
                logger.info(f"  SGI-Bench metrics: PassAll@5={extra_metrics.get('PassAll@5', 0):.3f}, PassAll@3={extra_metrics.get('PassAll@3', 0):.3f}, PassAll@1={extra_metrics.get('PassAll@1', 0):.3f}, SER={extra_metrics.get('SER', 0):.3f}")
            elif "wetexperiment" in dataset_lower:
                logger.info(f"  SGI-Bench metrics: action_sim={extra_metrics.get('action_sequence_similarity', 0):.3f}, param_acc={extra_metrics.get('parameter_accuracy', 0):.3f}, final={extra_metrics.get('final_score', 0):.3f}")
            elif "ideageneration" in dataset_lower:
                logger.info(f"  SGI-Bench metrics: final={extra_metrics.get('final_score', 0):.1f}, eff={extra_metrics.get('effectiveness', 0):.1f}, nov={extra_metrics.get('novelty', 0):.1f}, det={extra_metrics.get('detailedness', 0):.1f}, fea={extra_metrics.get('feasibility', 0):.1f}")
        else:
            # 通用输出
            if extra_metrics:
                metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in extra_metrics.items())
                logger.info(f"  Extra metrics: {metrics_str}")

