"""K12-Bench: drive the benchmark's own ``eval/eval_multiselect.py``.

K12-Bench measures curriculum cognition through five relation families, each
split into subtasks: concept grounding, prerequisite relations, neighbourhood
relations, supporting evidence and curriculum localization.  Every subtask is a
strict multi-select task, and the official scorer reports exact match (EM) and
F1 per family and overall.

This wrapper writes one model YAML per served endpoint under the clone's
``eval/configs/models`` (greedy, thinking disabled, 512 max tokens - a letter-set
answer stops on its own, so the budget only caps models that ignore the thinking
flag), evaluates the nine subtask files in parallel round-robin over the
endpoints, then re-opens the completed predictions to aggregate a single
``summary.json`` without issuing duplicate requests.  The temporary YAMLs are
always removed, including on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..metrics import build

NAME = "K12-Bench"
DEFAULT_MAX_TOKENS = 512

#: Subtask file stem -> appendix table family.
FAMILIES = {
    "ground": "Ground",
    "prereq": "Prereq",
    "neighbor": "Neighbor",
    "evidence": "Evidence",
    "locate": "Locate",
}

SUBTASK_FILES = (
    "ground_subtask1.jsonl",
    "ground_subtask2.jsonl",
    "prereq_subtask1.jsonl",
    "prereq_subtask2.jsonl",
    "neighbor.jsonl",
    "evidence_subtask1.jsonl",
    "evidence_subtask2.jsonl",
    "locate_subtask1.jsonl",
    "locate_subtask2.jsonl",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k12-repo", default=None, help="K12-Bench release checkout")
    parser.add_argument(
        "--k12-data-dir", default=None, help="directory holding the nine subtask JSONL files"
    )
    parser.add_argument("--max-samples-per-file", type=int, default=0)


def model_yaml(base_url: str, model: str, max_tokens: int) -> str:
    return (
        "api:\n"
        f"  base_url: {base_url}\n  model: {model}\n  api_key: EMPTY\n"
        f"request:\n  temperature: 0.0\n  top_p: 1.0\n  max_tokens: {max_tokens}\n"
        "  timeout: 600\n  extra_body:\n    chat_template_kwargs:\n"
        "      enable_thinking: false\n"
    )


def _run(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(argv) + "\n")
        stream.flush()
        proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}; log={log_path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo = (
        Path(
            args.k12_repo
            or getattr(args, "official", {}).get(NAME)
            or Path(args.data_root) / "k12_bench"
        )
        .expanduser()
        .resolve()
    )
    data_dir = Path(args.k12_data_dir or Path(args.data_root) / "k12_bench").expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = str(args.python)

    eval_script = repo / "eval" / "eval_multiselect.py"
    if not eval_script.is_file():
        raise SystemExit(f"official K12-Bench evaluation script not found: {eval_script}")
    if not data_dir.is_dir():
        raise SystemExit(f"K12-Bench evaluation data directory not found: {data_dir}")

    max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS
    models_dir = repo / "eval" / "configs" / "models"
    log_dir = output / "logs"
    stems: list[str] = []
    for index, url in enumerate(args.base_urls):
        # The model YAML lives inside the *shared* clone, so the stem has to be
        # unique per process: two concurrent runs would otherwise overwrite and
        # then delete each other's YAML.
        stem = f"automation_{os.getpid()}_{index}"
        stems.append(stem)
        (models_dir / f"{stem}.yaml").write_text(
            model_yaml(url, args.model, max_tokens), encoding="utf-8"
        )

    base = [python, str(eval_script)]
    common = ["-i", str(data_dir), "-o", str(output), "--concurrency", str(args.concurrency)]
    if args.max_samples_per_file:
        common += ["--max-samples-per-file", str(args.max_samples_per_file)]

    try:
        jobs = []
        for index, filename in enumerate(SUBTASK_FILES):
            stem = stems[index % len(stems)]
            jobs.append(
                (
                    [*base, "--model", stem, *common, "--glob", filename],
                    log_dir / f"{Path(filename).stem}.log",
                )
            )
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_run, argv, log) for argv, log in jobs]
            for future in futures:
                future.result()
        # Reference pass: re-open the completed predictions and aggregate them
        # into a single summary.json without re-issuing any request.
        _run(
            [
                *base,
                "--model",
                stems[0],
                "-i",
                str(data_dir),
                "-o",
                str(output),
                "--concurrency",
                "1",
            ],
            log_dir / "summary.log",
        )
    finally:
        for stem in stems:
            (models_dir / f"{stem}.yaml").unlink(missing_ok=True)

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    report = {
        "benchmark": "K12-Bench",
        "model": args.model,
        "files": len(SUBTASK_FILES),
        "overall": summary.get("overall", {}),
        "subtasks": summary.get("subtasks") or summary.get("files") or summary,
        "protocol": "official eval_multiselect.py; greedy; thinking off; official EM/F1",
    }
    report["metrics"] = _metrics(report)
    return report


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    """Merge the per-subtask rows into the appendix's per-family EM/F1."""
    totals: dict[str, list[float]] = {}
    for row in report.get("subtasks") or []:
        if not isinstance(row, dict):
            continue
        family = FAMILIES.get(Path(str(row.get("file", ""))).stem.split("_")[0])
        if family is None:
            continue
        count = float(row.get("count") or 0)
        bucket = totals.setdefault(family, [0.0, 0.0, 0.0])  # weight, EM, F1
        bucket[0] += count
        bucket[1] += count * float(row.get("exact_match") or 0)
        bucket[2] += count * float(row.get("f1") or 0)
    values: dict[str, float | None] = {}
    for family, (weight, exact, f1) in totals.items():
        values[f"{family} EM"] = exact / weight * 100 if weight else None
        values[f"{family} F1"] = f1 / weight * 100 if weight else None
    overall = report.get("overall") or {}
    values["Overall EM"] = _pct(overall.get("exact_match"))
    values["Overall F1"] = _pct(overall.get("f1"))
    return build("K12-Bench", values)


def _pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100
