"""Reading, field mapping and pool writing shared by the stage 2 cleaning scripts.

Source snapshots live under the matching stage 1 capability directory (`huggingface/`, `github/`,
`direct/`); a cleaning script normalizes them and writes `<dataset>/cleaned/kept.jsonl`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import pools

LETTERS = "ABCDEFGH"


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read source snapshots from parquet, jsonl or a JSON list."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        yield from pq.read_table(path).to_pylist()
        return
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or []
        yield from data
        return
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_snapshot(*paths: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Recursively read every snapshot file, yielding the source path with each row."""
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if file.is_file() and file.suffix in {".parquet", ".jsonl", ".json"}:
                for row in iter_rows(file):
                    yield file, row


def pick(row: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Return the first non-empty field among aliases."""
    for key in keys:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def options_text(options: Any) -> str:
    """Render options as "A. xx\\nB. yy"."""
    if not options:
        return ""
    if isinstance(options, dict):
        texts = list(options.get("text") or [])
        labels = list(options.get("label") or [])
        if texts:
            return "\n".join(
                f"{labels[i] if i < len(labels) else LETTERS[i]}. {text}" for i, text in enumerate(texts)
            )
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        parts = []
        for index, item in enumerate(options):
            if isinstance(item, dict):
                label = str(item.get("label") or LETTERS[index])
                parts.append(f"{label}. {item.get('text', '')}")
            else:
                parts.append(f"{LETTERS[index]}. {item}")
        return "\n".join(parts)
    return str(options)


def option_map(options: Any) -> dict[str, str]:
    """Return label → option text."""
    mapping: dict[str, str] = {}
    if isinstance(options, dict):
        texts = list(options.get("text") or [])
        labels = list(options.get("label") or [])
        if texts:
            for index, text in enumerate(texts):
                mapping[str(labels[index] if index < len(labels) else LETTERS[index])] = str(text)
            return mapping
        return {str(key): str(value) for key, value in options.items()}
    if isinstance(options, list):
        for index, item in enumerate(options):
            if isinstance(item, dict):
                mapping[str(item.get("label") or LETTERS[index])] = str(item.get("text", ""))
            else:
                mapping[LETTERS[index]] = str(item)
    return mapping


def answer_letter(row: dict[str, Any], options: Any) -> str:
    """Normalize an answer to its option letter, falling back to the raw value."""
    answer = str(pick(row, "answer", "answerKey", "label", "correct", "gold")).strip()
    mapping = option_map(options)
    if answer.upper() in mapping:
        return answer.upper()
    for label, text in mapping.items():
        if text and text.strip() == answer:
            return label.upper()
    return answer


def record(
    dataset: str,
    uid: str,
    problem: str,
    *,
    options: Any = None,
    answer: str = "",
    answer_text: str = "",
    cot: str = "",
    passage: str = "",
    meta: dict[str, Any] | None = None,
    image_ref: Any = None,
    flags: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a unified record, omitting empty fields."""
    row: dict[str, Any] = {"uid": uid, "dataset": dataset, "problem": problem}
    optional = {
        "options": options_text(options),
        "answer": answer,
        "answer_text": answer_text,
        "cot": cot,
        "passage": passage,
        "meta": meta or {},
        "image_ref": image_ref,
    }
    for key, value in optional.items():
        if value not in (None, "", [], {}):
            row[key] = value
    if list(flags):
        row["flags"] = list(flags)
    return row


def write_pool(
    capability: str,
    dataset: str,
    rows: list[dict[str, Any]],
    *,
    removed: dict[str, list[dict[str, Any]]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write `<dataset>/cleaned/kept.jsonl` plus dropped rows and a report."""
    out = pools.cleaned_dir(capability, dataset)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "kept.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for reason, items in (removed or {}).items():
        with (out / f"removed_{reason}.jsonl").open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "dataset": dataset,
        "kept_rows": len(rows),
        "removed_by_reason": {reason: len(items) for reason, items in (removed or {}).items()},
    }
    if extra:
        report.update(extra)
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
