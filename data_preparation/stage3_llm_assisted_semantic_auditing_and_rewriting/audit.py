#!/usr/bin/env python3
"""Semantic auditing: per-example keep / rewrite / remove verdicts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from common import llm as client
import sources


def render(record: dict) -> str:
    keys = (
        "passage", "prompt", "problem", "question", "options", "answer", "answer_text",
        "cot", "essay", "scores", "feedback", "messages", "conversation", "hint",
        "student_answer", "student_attempt", "history", "annotation", "meta",
        "image_ref", "images", "context_messages", "candidate_teacher_turn",
    )
    lines = []
    for key in keys:
        value = record.get(key)
        if value in (None, "", [], {}):
            continue
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"[{key}]\n{value}")
    return "\n\n".join(lines)


def checks_of(obj: dict) -> dict:
    nested = obj.get("checks")
    if isinstance(nested, dict) and nested:
        return {str(k): v for k, v in nested.items()}
    return {
        k: v
        for k, v in obj.items()
        if k not in ("reason", "usability_score", "confidence", "issues", "summary")
    }


def _yes(value) -> bool:
    return str(value).strip().lower() in ("yes", "true")


CRITICAL_CHECK = "input_and_reference_valid"


def decide(record: dict, obj: dict, dimensions: tuple[str, ...]) -> str:
    checks = checks_of(obj)
    try:
        score = float(obj.get("usability_score"))
    except (TypeError, ValueError):
        return "rewrite"
    if not _yes(checks.get(CRITICAL_CHECK, "yes")) or score < 50:
        return "remove"
    if score >= 85 and all(_yes(checks.get(key, "no")) for key in dimensions):
        return "keep"
    return "rewrite"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pool", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    client.add_common_args(parser)
    args = parser.parse_args()

    pool = args.pool or sources.pool_path(args.dataset)
    out = args.out or (sources.audit_dir(args.dataset) / "audit" / "verdicts.jsonl")
    dimensions = sources.dimensions(args.dataset)

    done = client.load_done(out, "uid", lambda row: row.get("verdict") in ("keep", "rewrite", "remove"))
    records = []
    with pool.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            uid = str(record.get("uid") or f"{args.dataset}-{index}")
            record.setdefault("uid", uid)
            if uid in done:
                continue
            records.append(record)
            if args.limit and len(records) >= args.limit:
                break
    print(f"dataset={args.dataset} pool={pool} todo={len(records)} skip={len(done)}", flush=True)

    stats = asyncio.run(
        client.run(
            records,
            out,
            lambda record: (sources.system_prompt(args.dataset, record), render(record)),
            lambda record, obj, error: {
                "uid": record["uid"],
                "dataset": args.dataset,
                "verdict": decide(record, obj, dimensions) if obj else None,
                "usability_score": (obj or {}).get("usability_score"),
                "checks": checks_of(obj or {}),
                "dimensions": list(dimensions),
                "reason": str((obj or {}).get("reason", ""))[:300],
                "error": error,
            },
            chat=client.from_args(args),
            concurrency=args.concurrency,
        )
    )
    print({"dataset": args.dataset, "output": str(out), "stats": stats}, flush=True)


if __name__ == "__main__":
    main()
