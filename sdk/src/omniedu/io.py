"""JSONL helpers and request normalisation for batch runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Request:
    """One normalised batch item."""

    id: str
    messages: list[dict[str, Any]]
    images: list[Any] = field(default_factory=list)
    task: str | None = None
    system: str | None = None


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield non-empty JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one record, flushed immediately so an interrupted run keeps rows."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def compact_jsonl(path: str | Path, *, key: str, keep: Iterable[str]) -> int:
    """Rewrite ``path`` keeping at most one row per ``key``, in file order.

    Only rows whose key is in ``keep`` survive and the last row for a key wins.
    Returns the number of rows removed (0 when the file was already compact).
    """
    target = Path(path)
    if not target.exists():
        return 0
    keep_set = {str(item) for item in keep}
    rows: dict[str, dict[str, Any]] = {}
    total = 0
    for record in iter_jsonl(target):
        total += 1
        identifier = str(record.get(key))
        if identifier in keep_set:
            rows[identifier] = record
    if len(rows) == total:
        return 0
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in rows.values():
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    os.replace(temporary, target)
    return total - len(rows)


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, "", []):
            return value
    return None


def normalise_request(
    row: dict[str, Any],
    *,
    index: int = 0,
    task: str | None = None,
    system: str | None = None,
) -> Request:
    """Accept the shapes people actually have and return one canonical Request.

    Supported inputs:

    * ShareGPT rows: ``{"id", "messages": [...], "images": [...]}``
    * flat rows: ``{"id", "text" | "prompt" | "question", "image" | "images"}``

    ``<image>`` markers survive normalisation and are resolved when the messages
    are built.
    """
    identifier = _first(row, "id", "uid", "sample_id", "_sample_id", "key")
    if identifier is None:
        identifier = f"row-{index:06d}"

    messages = row.get("messages")
    request_system = _first(row, "system", "system_prompt")
    if messages:
        cleaned = []
        for message in messages:
            message = dict(message)
            if message.get("role") == "system":
                request_system = request_system or str(message.get("content") or "")
                continue
            cleaned.append(message)
        messages = cleaned
    else:
        text = _first(row, "text", "prompt", "question", "instruction")
        if text is None:
            raise ValueError(f"row {identifier!r} has neither `messages` nor a text field")
        messages = [{"role": "user", "content": str(text)}]

    images = _first(row, "images", "image", "img", "image_path")
    if images is None:
        images = []
    elif not isinstance(images, list):
        images = [images]

    return Request(
        id=str(identifier),
        messages=messages,
        images=images,
        task=_first(row, "task") or task,
        system=request_system or system,
    )
