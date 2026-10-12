"""JSONL helpers and the resume rule shared by every runner.

The rule implemented by :func:`resume` is the one the runners had grown
individually: a row is only "done" when its previous attempt succeeded, and the
output file is compacted before a retry so one sample never ends up on two
lines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects in ``path``, skipping unparsable lines."""
    with Path(path).open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, record: dict[str, Any], **dumps: Any) -> None:
    """Append one record and flush, so an interrupted run keeps everything."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", errors="ignore") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, **dumps) + "\n")
        handle.flush()


def compact_jsonl(path: str | Path, key: str, keep: Iterable[str], **dumps: Any) -> int:
    """Rewrite ``path`` keeping one row per key, in file order (last wins).

    Only rows whose key is in ``keep`` survive.  Returns the number of rows
    removed, so a caller can log the compaction.
    """
    target = Path(path)
    if not target.exists():
        return 0
    wanted = {str(item) for item in keep}
    rows: dict[str, dict[str, Any]] = {}
    total = 0
    for record in iter_jsonl(target):
        total += 1
        identifier = str(record.get(key))
        if identifier in wanted:
            rows[identifier] = record
    if len(rows) == total:
        return 0
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", errors="ignore") as handle:
        for record in rows.values():
            handle.write(json.dumps(record, ensure_ascii=False, **dumps) + "\n")
    os.replace(temporary, target)
    return total - len(rows)


def resume(
    path: str | Path,
    *,
    key: str,
    failed: Callable[[dict[str, Any]], bool] | None = None,
    **dumps: Any,
) -> set[str]:
    """Return the keys that already finished successfully.

    ``failed`` defaults to "the record carries a truthy ``error``".  Passing a
    custom predicate is how a benchmark expresses its own failure marker (for
    example the ``__ERROR__:`` prefix some runners put in the response field).
    Duplicate and failed rows are dropped from the file on the way out, so the
    caller can simply append new results.
    """
    if failed is None:

        def failed(record: dict[str, Any]) -> bool:  # type: ignore[misc]
            return bool(record.get("error"))

    target = Path(path)
    if not target.exists():
        return set()

    previous: dict[str, dict[str, Any]] = {}
    lines = 0
    for record in iter_jsonl(target):
        lines += 1
        previous[str(record.get(key))] = record
    done = {name for name, record in previous.items() if not failed(record)}
    if len(done) != lines:
        compact_jsonl(target, key, done, **dumps)
        print(f"[resume] compacted {target}: {lines} -> {len(done)} records", flush=True)
    return done
