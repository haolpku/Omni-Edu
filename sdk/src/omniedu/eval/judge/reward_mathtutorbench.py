"""Thin launcher around MathTutorBench's official reward-model implementation.

The scoring classes and ``evaluate_preference_accuracy`` are imported unchanged
from the official benchmark snapshot.  This wrapper only supplies a local model
path, an output directory, and the results YAML file the upstream script expects.
It is run with the interpreter that has ``torch`` (``--reward-python``), which is
usually not the one running ``omniedu eval``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
import yaml

# One variable-length full-set input reproducibly triggers a divide error in
# libcublasLt on some hosts.  Force PyTorch's supported cuBLAS backend before the
# reward model is loaded; this changes only the GEMM implementation.
try:
    torch.backends.cuda.preferred_blas_library("cublas")
except (AttributeError, RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-script", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()

    script = Path(args.official_script).resolve()
    data = Path(args.data_path).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Upstream updates ./results/results-<model>.yaml after scoring.
    stem = data.stem.removeprefix("generations-")
    model_name, task = stem.rsplit("-", 1)
    work = out / "official_work"
    (work / "results").mkdir(parents=True, exist_ok=True)
    result_yaml = work / "results" / f"results-{model_name}.yaml"
    if not result_yaml.exists():
        result_yaml.write_text(yaml.safe_dump({task: {}}))

    spec = importlib.util.spec_from_file_location("official_mathtutor_reward", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The released scorer hard-codes bfloat16; allow a float32 compatibility run
    # on hosts where the CUDA bf16 cublasLt kernel raises SIGFPE.
    if args.torch_dtype == "float32":
        original = module.AutoModelForSequenceClassification.from_pretrained

        def from_pretrained_float32(*call_args: object, **call_kwargs: object) -> object:
            call_kwargs["torch_dtype"] = torch.float32
            return original(*call_args, **call_kwargs)

        module.AutoModelForSequenceClassification.from_pretrained = from_pretrained_float32

    # The official function references the module-global args only for YAML naming.
    module.args = argparse.Namespace(data_path=str(data))
    os.chdir(work)
    result = module.evaluate_preference_accuracy(
        model_name=str(Path(args.model_path).resolve()),
        data_path=str(data),
        batch_size=args.batch_size,
        output_dir=str(out / "scores"),
    )
    summary = {
        "official_script": str(script),
        "official_reward_model": str(Path(args.model_path).resolve()),
        "data_path": str(data),
        "task": task,
        "model": model_name,
        "result": result,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
