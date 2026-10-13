#!/usr/bin/env python3
"""Filter candidates by the rubric thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools as common_pools

import rubrics

HERE = Path(__file__).resolve().parent
POOLS = HERE / "pools"
RESULTS = HERE / "results"
SCORES = RESULTS / "scores.jsonl"
OUT_DIR = RESULTS / "quality_filtered_pools"


def read_scores() -> dict[str, dict]:
    scores: dict[str, dict] = {}
    if not SCORES.exists():
        raise SystemExit(f"missing {SCORES}; run score.py first")
    with SCORES.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scores"):
                scores[row["record_id"]] = row
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=common_pools.CAPABILITIES, default=None)
    args = parser.parse_args()

    scores = read_scores()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for category in common_pools.CAPABILITIES:
        if args.category and category != args.category:
            continue
        pool = POOLS / f"{category}.jsonl"
        if not pool.is_file():
            continue
        kept: list[dict] = []
        dropped: Counter[str] = Counter()
        with pool.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = f"{record.get('category')}::{record.get('source')}::{record.get('uid', index)}"
                row = scores.get(record_id)
                if not row:
                    dropped["unscored"] += 1
                    continue
                if rubrics.keeps(row["family"], row["scores"], str(record.get("source"))):
                    kept.append(record)
                else:
                    dropped["below_rubric_threshold"] += 1
        with (OUT_DIR / f"{category}.jsonl").open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary[category] = {
            "kept_rows": len(kept),
            "dropped": dict(dropped),
        }
    (RESULTS / "filter_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
