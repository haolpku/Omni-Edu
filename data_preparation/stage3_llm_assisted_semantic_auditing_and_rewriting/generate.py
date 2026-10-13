#!/usr/bin/env python3
"""Generation tasks on the shared client (LongTutor synthetic, Eedi rationales)."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import audit
from common import llm as client
import sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(sources.GENERATE_OF))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--field", default="generated", help="column to write the generation into")
    parser.add_argument("--limit", type=int, default=0)
    client.add_common_args(parser)
    args = parser.parse_args()

    out = args.out or (args.input.parent / sources.GENERATE_OF[args.task] / "generated.jsonl")
    system = (sources.PROMPTS / f"{sources.GENERATE_OF[args.task]}.txt").read_text(
        encoding="utf-8"
    ).strip()

    done = client.load_done(out, "uid", lambda row: bool(row.get(args.field)))
    records = []
    with args.input.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            record.setdefault("uid", f"{args.task}-{index}")
            if str(record["uid"]) in done:
                continue
            records.append(record)
            if args.limit and len(records) >= args.limit:
                break
    print(f"task={args.task} todo={len(records)} skip={len(done)} out={out}", flush=True)

    asyncio.run(
        client.run(
            records,
            out,
            lambda record: (system, audit.render(record)),
            lambda record, obj, error: {
                "uid": record["uid"],
                "dataset": record.get("dataset", args.task),
                args.field: (obj or {}).get(args.field, ""),
                "error": error,
            },
            chat=client.from_args(args),
            concurrency=args.concurrency,
        )
    )
    print({"task": args.task, "output": str(out)}, flush=True)


if __name__ == "__main__":
    main()
