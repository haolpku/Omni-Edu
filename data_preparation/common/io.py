"""JSON / JSONL readers and writers shared by every stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one object per non-empty line of ``path``."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a whole JSONL file into a list."""
    return list(iter_jsonl(path))


def write_jsonl(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    sort_keys: bool = False,
) -> int:
    """Write ``rows`` as JSONL and return how many records were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=sort_keys) + "\n")
            count += 1
    return count


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    """Write one JSON document with two-space indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
