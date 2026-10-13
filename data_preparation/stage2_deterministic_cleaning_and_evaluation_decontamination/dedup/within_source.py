#!/usr/bin/env python3
"""Within-dataset deduplication (one row per exact duplicate)."""

from __future__ import annotations

import argparse
import json
import re
import sys as _sys
import unicodedata
from pathlib import Path

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

DEDUP_LAYER = "intra_dataset"
REPORT_KEYS = ("dataset", "family", "input", "input_rows", "kept_rows", "removed_rows")

FAMILY = {
    "gsm8k-main": "math",
    "hendrycks-math": "math",
    "mathqa": "math",
    "cmm-math": "math",
    "SciInstruct (CN math)": "math",
    "openr1-default": "math",
    "scienceqa": "math",
    "geometry3k": "math",
    "arc": "short_answer",
    "qasc": "short_answer",
    "sciq": "short_answer",
    "cjeval": "short_answer",
    "ai2d": "short_answer",
    "race": "reading",
    "RACE": "reading",
    "C3": "reading",
    "TQA (text)": "reading",
    "TQA (diagram)": "reading",
    "essay-asap": "dialogue",
    "essay-csee": "dialogue",
    "essay-ellipse": "dialogue",
}

KEEP_SYMBOLS = set("<>=+-*/^\\{}")


def norm_symbols(value) -> str:
    text = str(value).lower().replace("$", "")
    return unicodedata.normalize(
        "NFKC", "".join(c for c in text if c.isalnum() or c in KEEP_SYMBOLS)
    )


def norm_words(value) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\W+", "", text)


def messages_of(row: dict):
    for key in ("messages", "conversation", "dialog", "dialog_history"):
        value = row.get(key)
        if value:
            return value
    return None


def dedup_key(row: dict, family: str):
    if family == "math":
        return ("math", norm_symbols(row.get("problem")), norm_symbols(row.get("options")))
    if family == "short_answer":
        return (
            "short_answer",
            norm_words(row.get("problem") or row.get("question")),
            norm_words(row.get("student_answer") or row.get("student_response")),
            norm_words(row.get("label") or row.get("label_5way") or row.get("answer")),
        )
    if family == "reading":
        return (
            "reading",
            norm_words(row.get("passage")),
            norm_words(row.get("problem") or row.get("question")),
            norm_words(row.get("options")),
        )
    if family == "dialogue":
        messages = messages_of(row)
        if messages:
            return ("dialogue", json.dumps(messages, ensure_ascii=False, sort_keys=True))
    return (
        "generic",
        norm_words(row.get("problem") or row.get("question") or row.get("instruction")),
        norm_words(row.get("options")),
        norm_words(row.get("answer")),
    )


def cot_length(row: dict) -> int:
    for key in ("cot", "feedback", "response", "solution", "messages"):
        value = row.get(key)
        if value:
            return len(json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value)
    return 0


def dataset_dirs(pool_root: Path | None = None) -> dict[str, Path]:
    return {name: path.parent.parent for name, path in pools.find_pools(pool_root).items()}


def input_path(dataset_dir: Path) -> Path:
    return dataset_dir / pools.STANDARD_POOL


def dedup_dataset(dataset: str, dataset_dir: Path, source: Path, dry_run: bool) -> dict:
    family = FAMILY.get(dataset, "generic")
    rows = [json.loads(line) for line in source.open(encoding="utf-8-sig") if line.strip()]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(dedup_key(row, family), []).append(row)

    kept: list[dict] = []
    removed: list[dict] = []
    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue
        best = max(range(len(group)), key=lambda i: cot_length(group[i]))
        for index, row in enumerate(group):
            if index == best:
                kept.append(row)
            else:
                duplicate = dict(row)
                duplicate.update(
                    _reason="intra_dataset_duplicate",
                    _kept_uid=group[best].get("uid") or group[best].get("record_id"),
                    _dedup_layer=DEDUP_LAYER,
                )
                removed.append(duplicate)

    report = {
        "dataset": dataset,
        "family": family,
        "input": str(source),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
    }
    if not dry_run:
        out_dir = dataset_dir / "dedup" / "within"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows_to_write in (
            ("kept.jsonl", kept),
            ("removed_intra_duplicates.jsonl", removed),
        ):
            with (out_dir / name).open("w", encoding="utf-8") as handle:
                for row in rows_to_write:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pool", type=Path, default=None, help="process a single dataset directory")
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=None,
        help="root holding the datasets (defaults to the stage 2 directory, same layout required)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    parser.add_argument("--list-only", action="store_true", help="list the discovered datasets only")
    args = parser.parse_args()

    datasets = (
        {args.pool.name: args.pool} if args.pool is not None else dataset_dirs(args.pool_root)
    )
    if not datasets:
        raise SystemExit(
            "no dataset pool found: every dataset must publish its cleaning output to "
            "<dataset>/cleaned/kept.jsonl (see data_preparation/common/pools.py)"
        )

    if args.list_only:
        for dataset in sorted(datasets):
            print(f"{dataset:<40} {FAMILY.get(dataset, 'generic'):<13} {input_path(datasets[dataset])}")
        print(f"datasets discovered: {len(datasets)}")
        return

    missing = [name for name, directory in datasets.items() if not input_path(directory).is_file()]
    if missing:
        raise SystemExit(f"missing cleaned pool for: {', '.join(sorted(missing))}")

    reports = [
        dedup_dataset(dataset, datasets[dataset], input_path(datasets[dataset]), args.dry_run)
        for dataset in sorted(datasets)
    ]
    for report in reports:
        print(
            "{dataset:<36} family={family:<12} in={input_rows:>7} kept={kept_rows:>7} removed={removed_rows:>6}"
            .format(**report)
        )
    print(f"datasets processed: {len(reports)}")


if __name__ == "__main__":
    main()
