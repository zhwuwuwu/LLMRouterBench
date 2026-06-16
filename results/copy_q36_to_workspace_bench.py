"""Copy qwen3.6-35b (FP16) and qwen3.6-35b-awq-4bit re-eval'd / original results
into D:/workspace/results/bench/<ds>/<split>/<model>/ (incremental add, preserve existing).

For 12 reeval'd datasets: copy the -reeval.json only.
For 5 non-reeval'd datasets (aime, math500, mathbench, livemathbench, finqa): copy original.
"""
import shutil
from pathlib import Path

SRC_BENCH = Path("D:/router/LLMRouterBench/results/bench")
DST = Path("D:/workspace/results/bench")

MODELS = {
    "qwen3.6-35b": Path("D:/router/LLMRouterBench/results/qwen36_35b_re_eval"),
    "qwen3.6-35b-awq-4bit": Path("D:/router/LLMRouterBench/results/qwen36_35b_awq_4bit_re_eval"),
}

DATASETS = [
    ("aime", "hybrid"), ("math500", "test"), ("mathbench", "test"),
    ("livemathbench", "test"), ("finqa", "test"),
    ("humaneval", "test"), ("mbpp", "test"), ("livecodebench", "test"),
    ("kandk", "test"), ("bbh", "test"),
    ("arcc", "test"), ("gpqa", "test"), ("medqa", "test"),
    ("mmlupro", "test_3000"), ("winogrande", "valid"),
    ("emorynlp", "test"), ("meld", "test"),
]

REEVAL_DATASETS = {"humaneval", "mbpp", "livecodebench", "kandk", "bbh",
                   "arcc", "gpqa", "medqa", "mmlupro", "winogrande",
                   "emorynlp", "meld"}

NON_REEVAL = {"aime", "math500", "mathbench", "livemathbench", "finqa"}

assert REEVAL_DATASETS | NON_REEVAL == {d for d, _ in DATASETS}

copied = 0
for model, reeval_dir in MODELS.items():
    print(f"\n=== {model} ===")
    for ds, split in DATASETS:
        dst_dir = DST / ds / split / model
        dst_dir.mkdir(parents=True, exist_ok=True)

        if ds in REEVAL_DATASETS:
            # Find -reeval.json in reeval dir
            matches = list(reeval_dir.glob(f"{ds}-{split}-{model}-*reeval.json")) + \
                      list(reeval_dir.glob(f"{ds}-{split}-{model}-reeval.json"))
            matches = sorted(set(matches))
            assert len(matches) == 1, f"expected 1 reeval for {ds}/{model}, got {matches}"
            src = matches[0]
        else:
            src_dir = SRC_BENCH / ds / split / model
            files = sorted(src_dir.glob("*.json"))
            assert len(files) == 1, f"expected 1 orig for {ds}/{model}, got {files}"
            src = files[0]

        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        copied += 1
        print(f"  [OK] {ds:<16} {split:<10} -> {dst.name}")

print(f"\nDone. Copied {copied} files.")
