#!/usr/bin/env python3
"""Rewrite the supervision of verdict=rewrite rows, then re-audit them."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import audit
from common import llm as client
import sources

REWRITE_RULE = (
    "\nRewrite the supervision only: keep the input (passage/question/essay/dialogue) unchanged, "
    'output one line JSON {"rewritten": "<the corrected supervision>", "changed": "yes|no", '
    '"reason": "<short>"}. Do not output any thinking process.'
)


def load_records(pool: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with pool.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            records[str(record.get("uid") or index)] = record
    return records


def load_verdicts(path: Path) -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    if not path.exists():
        return verdicts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("verdict"):
                verdicts[str(row["uid"])] = row
    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--limit", type=int, default=0)
    client.add_common_args(parser)
    args = parser.parse_args()

    pool = sources.pool_path(args.dataset)
    audit_dir = sources.audit_dir(args.dataset) / "audit"
    rewrite_dir = sources.audit_dir(args.dataset) / "rewrite"
    rewrite_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(pool)
    verdicts = load_verdicts(audit_dir / "verdicts.jsonl")
    todo = [records[uid] for uid, row in verdicts.items() if row["verdict"] == "rewrite" and uid in records]
    if args.limit:
        todo = todo[: args.limit]
    rewritten_path = rewrite_dir / "rewritten.jsonl"
    done = client.load_done(rewritten_path, "uid", lambda row: bool(row.get("rewritten")))
    todo = [record for record in todo if record["uid"] not in done]
    print(f"dataset={args.dataset} rewrite_todo={len(todo)}", flush=True)

    system = sources.prompt_text(args.dataset) + REWRITE_RULE
    asyncio.run(
        client.run(
            todo,
            rewritten_path,
            lambda record: (system, audit.render(record)),
            lambda record, obj, error: {
                "uid": record["uid"],
                "dataset": args.dataset,
                "rewritten": (obj or {}).get("rewritten", ""),
                "changed": (obj or {}).get("changed", ""),
                "reason": str((obj or {}).get("reason", ""))[:300],
                "error": error,
            },
            chat=client.from_args(args),
            concurrency=args.concurrency,
        )
    )

    rewritten = {}
    with rewritten_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("rewritten"):
                    rewritten[str(row["uid"])] = row["rewritten"]
    reaudit_records = []
    for uid, text in rewritten.items():
        record = dict(records[uid])
        for key in ("cot", "feedback", "answer_text", "hint"):
            if key in record:
                record[key] = text
                break
        else:
            record["cot"] = text
        reaudit_records.append(record)
    dimensions = sources.dimensions(args.dataset)
    reaudit_path = rewrite_dir / "reaudit.jsonl"
    asyncio.run(
        client.run(
            reaudit_records,
            reaudit_path,
            lambda record: (sources.system_prompt(args.dataset, record), audit.render(record)),
            lambda record, obj, error: {
                "uid": record["uid"],
                "dataset": args.dataset,
                "verdict": audit.decide(record, obj, dimensions) if obj else None,
                "usability_score": (obj or {}).get("usability_score"),
                "checks": audit.checks_of(obj or {}),
                "reason": str((obj or {}).get("reason", ""))[:300],
                "error": error,
            },
            chat=client.from_args(args),
            concurrency=args.concurrency,
        )
    )
    print({"dataset": args.dataset, "rewritten": len(rewritten), "reaudit": str(reaudit_path)}, flush=True)


if __name__ == "__main__":
    main()
