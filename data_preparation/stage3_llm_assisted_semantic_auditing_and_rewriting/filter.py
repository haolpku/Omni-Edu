#!/usr/bin/env python3
"""Filter a pool by the audit verdicts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sources


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    pool = sources.pool_path(args.dataset)
    base = sources.audit_dir(args.dataset)
    records = {str(row.get("uid")): row for row in read_jsonl(pool)}

    verdicts = {str(row["uid"]): row for row in read_jsonl(base / "audit" / "verdicts.jsonl") if row.get("verdict")}
    for row in read_jsonl(base / "rewrite" / "reaudit.jsonl"):
        if row.get("verdict"):
            verdicts[str(row["uid"])] = row

    kept, removed = [], []
    for uid, record in records.items():
        verdict = verdicts.get(uid, {}).get("verdict")
        if verdict == "keep":
            kept.append(record)
        elif verdict == "remove":
            removed.append({**record, "_reason": "audit_remove"})

    write_jsonl(base / "audited" / "kept.jsonl", kept)
    write_jsonl(base / "audited" / "removed.jsonl", removed)
    report = {
        "dataset": args.dataset,
        "pool": str(pool),
        "input_rows": len(records),
        "kept_rows": len(kept),
        "removed_rows": len(removed),
        "pending_rewrite": sum(1 for row in verdicts.values() if row.get("verdict") == "rewrite"),
        "unscored": len(records) - len(verdicts),
    }
    (base / "audited" / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
