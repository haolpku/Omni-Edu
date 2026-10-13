#!/usr/bin/env python3
"""Cross-dataset deduplication across all education datasets."""

from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path

from within_source import FAMILY, dedup_key, dataset_dirs, input_path, norm_words

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

OUTPUT_DIR = pools.STAGE2 / "dedup" / "across_sources"

PRIORITY = {
    "math": ["gsm8k-main", "hendrycks-math", "cmm-math", "openr1-default", "mathqa", "SciInstruct (CN math)"],
    "short_answer": ["scienceqa", "ai2d", "arc", "qasc", "sciq", "cjeval"],
    "reading": ["race", "RACE", "C3", "TQA (text)", "TQA (diagram)"],
    "dialogue": ["tutorchat_education_dialogue", "socraticmath", "socrateach_convolearn", "essay-asap"],
}


def answer_of(row: dict) -> str:
    for key in ("answer", "answer_text", "label", "label_5way", "response", "output"):
        value = row.get(key)
        if value:
            return norm_words(value)
    return ""


def answers_match(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    return len(left) >= 3 and len(right) >= 3 and (left in right or right in left)


def order_datasets(datasets: list[str], family: str) -> list[str]:
    priority = PRIORITY.get(family, [])
    return sorted(datasets, key=lambda name: (priority.index(name) if name in priority else len(priority), name))


def run_family(family: str, members: dict[str, tuple[dict, Path]], dry_run: bool) -> dict:
    seen: dict[tuple, tuple[str, dict]] = {}
    kept: list[dict] = []
    removed: list[dict] = []
    per_dataset = {}

    for dataset in order_datasets(list(members), family):
        rows, source = members[dataset]
        kept_here = 0
        for row in rows:
            key = dedup_key(row, family)
            owner = seen.get(key)
            if owner is not None and answers_match(answer_of(owner[1]), answer_of(row)):
                duplicate = dict(row)
                duplicate.update(
                    _reason="cross_dataset_duplicate",
                    _kept_in=owner[0],
                    _kept_uid=owner[1].get("uid") or owner[1].get("record_id"),
                    _dedup_layer="cross_dedup",
                )
                removed.append(duplicate)
                continue
            seen.setdefault(key, (dataset, row))
            kept.append(row)
            kept_here += 1
        per_dataset[dataset] = {"input": str(source), "rows": len(rows), "kept": kept_here}

    if not dry_run:
        out_dir = OUTPUT_DIR / family
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "kept.jsonl").open("w", encoding="utf-8") as handle:
            for row in kept:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out_dir / "removed_cross_duplicates.jsonl").open("w", encoding="utf-8") as handle:
            for row in removed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out_dir / "report.json").write_text(
            json.dumps(
                {
                    "family": family,
                    "datasets": per_dataset,
                    "kept_rows": len(kept),
                    "removed_rows": len(removed),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "family": family,
        "datasets": len(members),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=None,
        help="root holding the datasets (defaults to the stage 2 directory, same layout required)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report the family distribution only")
    parser.add_argument("--list-only", action="store_true", help="list the discovered datasets only")
    args = parser.parse_args()

    datasets = dataset_dirs(args.pool_root)
    if not datasets:
        raise SystemExit(
            "no dataset pool found: every dataset must publish its cleaning output to "
            "<dataset>/cleaned/kept.jsonl (see data_preparation/common/pools.py)"
        )
    if args.list_only:
        for dataset in sorted(datasets):
            source = input_path(datasets[dataset])
            print(f"{dataset:<40} {FAMILY.get(dataset, 'generic'):<13} {source}")
        print(f"datasets discovered: {len(datasets)}")
        return
    families: dict[str, dict[str, tuple[dict, Path]]] = {}
    for dataset, dataset_dir in datasets.items():
        within = dataset_dir / pools.STANDARD_WITHIN_DEDUP
        source = within if within.is_file() else input_path(dataset_dir)
        family = FAMILY.get(dataset, "generic")
        rows = [json.loads(line) for line in source.open(encoding="utf-8-sig") if line.strip()]
        families.setdefault(family, {})[dataset] = (rows, source)

    reports = []
    for family in sorted(families):
        members = families[family]
        if args.dry_run:
            reports.append(
                {"family": family, "datasets": len(members), "kept_rows": 0, "removed_rows": 0}
            )
            print(f"{family:<14} datasets={len(members):>3}  " + ", ".join(sorted(members)))
            continue
        report = run_family(family, members, dry_run=False)
        reports.append(report)
        print(
            "{family:<14} datasets={datasets:>3} kept={kept_rows:>7} removed={removed_rows:>6}".format(**report)
        )
    print(f"families processed: {len(reports)}")


if __name__ == "__main__":
    main()
