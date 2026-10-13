"""MathTutorBench: official generation plus the official pedagogical reward model.

Generation drives the benchmark's own ``main.py`` for the four tasks
(scaffolding and pedagogy, each with an easy and a hard split).  Scoring loads
the benchmark authors' local pedagogical reward model through the thin launcher
that imports the released scoring classes unchanged.

Our task YAMLs and decontaminated datasets ship inside the package and are
copied into the clone before the run, so the generation always targets the
decontaminated inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..resources import mathtutorbench_data
from ..metrics import build

NAME = "MathTutorBench"
TASKS = (
    "scaffolding_generation",
    "pedagogy_following",
    "scaffolding_generation_hard",
    "pedagogy_following_hard",
)

#: task id -> appendix column.
WIN_RATE_COLUMNS = {
    "scaffolding_generation": "Scaffold WR",
    "scaffolding_generation_hard": "Scaffold-hard WR",
    "pedagogy_following": "Pedagogy WR",
    "pedagogy_following_hard": "Pedagogy-hard WR",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mathtutor-repo", default=None, help="MathTutorBench checkout")
    parser.add_argument("--reward-model", default=None, help="local pedagogical reward model path")
    parser.add_argument(
        "--skip-reward", action="store_true", help="generate only; defer the reward model"
    )
    parser.add_argument("--reward-cuda-visible-devices", default=None)
    parser.add_argument(
        "--mathtutor-tasks",
        default=None,
        help="comma-separated subset of the four tasks (default: all)",
    )


def _run(argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    child = {**os.environ, **(env or {})}
    proc = subprocess.run(
        [str(item) for item in argv],
        cwd=str(cwd) if cwd else None,
        env=child,
        text=True,
    )
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str(x) for x in argv)}")


def _sync_overrides(repo: Path) -> None:
    """Copy our task YAMLs and decontaminated datasets into the clone (idempotent).

    The configs keep upstream's ``datasets/...`` paths but point at the
    ``_clean`` bridge files, so the clean datasets land under ``datasets/``.
    """
    configs = mathtutorbench_data("configs")
    if configs.is_dir():
        shutil.copytree(configs, repo / "configs", dirs_exist_ok=True)
    datasets = mathtutorbench_data("clean_datasets")
    if datasets.is_dir():
        shutil.copytree(datasets, repo / "datasets", dirs_exist_ok=True)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = (
        Path(
            args.mathtutor_repo
            or getattr(args, "official", {}).get(NAME)
            or Path(args.data_root) / "mathtutorbench"
        )
        .expanduser()
        .resolve()
    )
    if not (repo / "main.py").is_file():
        raise SystemExit(
            f"MathTutorBench checkout not found under {repo}; set --official-root {NAME}=PATH"
        )
    _sync_overrides(repo)

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = (
        tuple(item.strip() for item in args.mathtutor_tasks.split(",") if item.strip())
        if args.mathtutor_tasks
        else TASKS
    )
    jobs = []
    for task in tasks:
        task_out = output / task
        task_out.mkdir(parents=True, exist_ok=True)
        jobs.append(
            [
                args.python,
                "main.py",
                "--tasks",
                task + ".yaml",
                "--provider",
                "completion_api",
                "--model_args",
                f"base_url={args.base_urls[0]},model={args.model},is_chat=True,is_thinking=True,"
                "temperature=0.0,max_tokens=512,max_retries=3",
                "--output",
                str(task_out),
            ]
        )

    def launch(argv: list[str]) -> None:
        _run(argv, cwd=repo)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(launch, argv) for argv in jobs]
        for future in futures:
            future.result()

    summary: dict[str, Any] = {"benchmark": "MathTutorBench", "model": args.model, "tasks": {}}
    if args.skip_reward:
        summary["reward"] = "deferred (--skip-reward)"
        summary["metrics"] = build("MathTutorBench", {})
        return summary

    reward_model = (
        args.reward_model
        or Path(args.data_root) / "mathtutorbench" / "Qwen2.5-1.5B-pedagogical-rewardmodel"
    )
    official_script = repo / "reward_model" / "compute_scaffolding_score.py"
    launcher = Path(__file__).resolve().parent.parent / "judge" / "reward_mathtutorbench.py"
    for task in tasks:
        sources = sorted((output / task).glob("generations-*.json"))
        if not sources:
            raise RuntimeError(f"MathTutorBench generation missing: {task}")
        reward_out = output / task / "official_reward"
        env = {
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DISABLE_ADDMM_CUDA_LT": "1",
        }
        if args.reward_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.reward_cuda_visible_devices
        _run(
            [
                args.reward_python,
                str(launcher),
                "--official-script",
                str(official_script),
                "--model-path",
                str(reward_model),
                "--data-path",
                str(sources[0]),
                "--output-dir",
                str(reward_out),
                "--batch-size",
                "1",
                "--torch-dtype",
                "bfloat16",
            ],
            env=env,
        )
        result = reward_out / "summary.json"
        if result.is_file():
            summary["tasks"][task] = json.loads(result.read_text(encoding="utf-8"))
    summary["protocol"] = "official main.py generation + official pedagogical reward model"
    summary["metrics"] = build(
        "MathTutorBench",
        {
            column: _win_rate(summary["tasks"].get(task))
            for task, column in WIN_RATE_COLUMNS.items()
        },
    )
    return summary


def _win_rate(reward: Any) -> float | None:
    value = ((reward or {}).get("result") or {}).get("win_rate")
    return None if value is None else float(value) * 100
