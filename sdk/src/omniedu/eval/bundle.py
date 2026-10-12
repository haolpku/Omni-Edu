"""Turn generated predictions into the workload each judge expects.

A judge input row is not a prediction row: the judge needs the question, the
reference answer and the model response under fixed keys, while a runner keeps
whatever is convenient.  Each builder here converts one benchmark's output into
the workload its judge consumes, written next to the run as
``<benchmark>/judge/<task>.jsonl`` so ``omniedu eval`` feeds ``omniedu judge``
directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def workload_path(output_dir: str | Path, task: str) -> Path:
    return Path(output_dir) / "judge" / task


def write_workload(output_dir: str | Path, task: str, rows: Iterable[dict[str, Any]]) -> Path:
    path = workload_path(output_dir, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
