"""Copy qwen3.6-35b-awq-4bit results to workspace bench_q36 dir.

For reeval'd datasets (humaneval, mbpp, kandk, livecodebench), copy only the reeval version
(renamed to match pattern without timestamp).
For other datasets, copy the original JSON.
"""
import shutil
from pathlib import Path

SRC_BENCH = Path("D:/router/LLMRouterBench/results/bench")
SRC_REEVAL = Path("D:/router/LLMRouterBench/results/qwen36_35b_awq_4bit_re_eval")
DST = Path("D:/workspace/results/bench_q36")
MODEL = "qwen3.6-35b-awq-4bit"

DATASETS = [
    ("aime", "hybrid"), ("math500", "test"), ("mathbench", "test"),
    ("livemathbench", "test"), ("humaneval", "test"), ("mbpp", "test"),
    ("livecodebench", "test"), ("bbh", "test"), ("kandk", "test"),
    ("mmlupro", "test_3000"), ("gpqa", "test"), ("finqa", "test"),
    ("medqa", "test"), ("arcc", "test"), ("winogrande", "valid"),
    ("emorynlp", "test"), ("meld", "test"),
]

REEVAL = {
    "humaneval": SRC_REEVAL / "humaneval-test-qwen3.6-35b-awq-4bit-reeval.json",
    "mbpp": SRC_REEVAL / "mbpp-test-qwen3.6-35b-awq-4bit-reeval.json",
    "kandk": SRC_REEVAL / "kandk-test-qwen3.6-35b-awq-4bit-20260420_193019-reeval.json",
    "livecodebench": SRC_REEVAL / "livecodebench-test-qwen3.6-35b-awq-4bit-20260420_112501-reeval.json",
}

for ds, split in DATASETS:
    dst_dir = DST / ds / split / MODEL
    dst_dir.mkdir(parents=True, exist_ok=True)
    if ds in REEVAL:
        src = REEVAL[ds]
        # Simplified name pattern matching FP16: {ds}-{split}-{model}-reeval.json
        dst_name = f"{ds}-{split}-{MODEL}-reeval.json"
    else:
        src_dir = SRC_BENCH / ds / split / MODEL
        files = sorted(src_dir.glob("*.json"))
        assert files, f"no file in {src_dir}"
        src = files[0]
        dst_name = src.name
    dst = dst_dir / dst_name
    shutil.copy2(src, dst)
    print(f"[OK] {ds:<16} -> {dst}")

print("\nDone.")
