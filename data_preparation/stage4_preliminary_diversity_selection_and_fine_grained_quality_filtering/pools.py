#!/usr/bin/env python3
"""Candidate pool assembly from the stage 3 audited pools."""

from __future__ import annotations

import argparse
import hashlib
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

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent / "stage3_llm_assisted_semantic_auditing_and_rewriting"
OUT_DIR = HERE / "pools"

KIND_OF = {
    "RACE": "reading",
    "C3": "reading",
    "TQA (text)": "reading",
    "TQA (diagram)": "reading",
    "ASAP 2.0": "essay",
    "CSEE": "essay",
    "ELLIPSE": "essay",
    "LongTutor-Gold": "longitudinal_tutoring",
    "LongTutor-Silver": "longitudinal_tutoring",
    "LongTutor-Synthetic": "longitudinal_tutoring",
    "MathFish-train (strict)": "strict_alignment_binary",
    "MathFish-train (multi-relation)": "multi_relation",
}


def audited_pools() -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for category in common_pools.CAPABILITIES:
        base = STAGE3 / category
        if not base.is_dir():
            continue
        for pool in sorted(base.rglob("audited/kept.jsonl")):
            found.append((category, pool.parent.parent.name, pool))
    return found


def payload_hash(record: dict) -> str:
    payload = record.get("payload") or {
        key: value for key, value in record.items() if key not in ("category", "source", "kind", "extra_info")
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def build(category: str, entries: list[tuple[str, str, Path]]) -> dict:
    kept: list[dict] = []
    seen: set[str] = set()
    per_source: Counter[str] = Counter()
    dropped = 0
    for _, source, path in entries:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = payload_hash(record)
                if digest in seen:
                    dropped += 1
                    continue
                seen.add(digest)
                record["category"] = category
                record["source"] = source
                record.setdefault("kind", KIND_OF.get(source, "qa"))
                if not record.get("cot") and record.get("answer"):
                    record["kind"] = "answer_only"
                kept.append(record)
                per_source[source] += 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{category}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "category": category,
        "sources": dict(sorted(per_source.items())),
        "kept_rows": len(kept),
        "cross_source_duplicates_dropped": dropped,
    }
    (OUT_DIR / f"{category}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=common_pools.CAPABILITIES, default=None)
    args = parser.parse_args()

    entries = audited_pools()
    if not entries:
        raise SystemExit(
            "no audited pool found: run stage3 audit/rewrite/filter first "
            "(expected stage3/<capability>/<dataset>/audited/kept.jsonl)"
        )
    for category in common_pools.CAPABILITIES:
        if args.category and category != args.category:
            continue
        selected = [entry for entry in entries if entry[0] == category]
        if not selected:
            continue
        print(build(category, selected), flush=True)


if __name__ == "__main__":
    main()
