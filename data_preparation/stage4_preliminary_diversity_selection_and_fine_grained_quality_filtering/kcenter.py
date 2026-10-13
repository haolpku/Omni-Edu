#!/usr/bin/env python3
"""Source-level k-center compression of over-represented sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))

HERE = Path(__file__).resolve().parent
POOLS = HERE / "pools"
OUT_DIR = HERE / "preliminary"

TARGETS = {
    "RACE": 5_000,
    "AquilaEdu": 2_000,
    "CJEval": 2_000,
    "OpenR1": 1_000,
    "SciInstruct (EN)": 1_000,
    "SciQ": 1_000,
    "MathQA": 1_000,
    "ASAP 2.0": 500,
    "CSEE": 500,
    "ELLIPSE": 500,
}


def target_text(record: dict) -> str:
    parts = [
        str(record.get(key) or "")
        for key in ("passage", "problem", "question", "prompt", "essay", "options", "answer", "cot", "feedback")
    ]
    return "\n".join(part for part in parts if part.strip())[:8000]


def greedy_kcenter(vectors: np.ndarray, k: int) -> list[int]:
    if k >= len(vectors):
        return list(range(len(vectors)))
    center = vectors.mean(axis=0, keepdims=True)
    first = int(np.argmin(((vectors - center) ** 2).sum(axis=1)))
    selected = [first]
    best = ((vectors - vectors[first]) ** 2).sum(axis=1)
    best[first] = -1.0
    while len(selected) < k:
        index = int(np.argmax(best))
        selected.append(index)
        best = np.minimum(best, ((vectors - vectors[index]) ** 2).sum(axis=1))
        best[index] = -1.0
    return selected


def load_source(source: str) -> list[dict]:
    records: list[dict] = []
    for pool in sorted(POOLS.glob("*.jsonl")):
        with pool.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("source") == source:
                    records.append(record)
    if not records:
        raise SystemExit(f"source {source!r} not found in {POOLS}/*.jsonl")
    return records


def compress(source: str, target: int, embed_model: Path) -> dict:
    from sentence_transformers import SentenceTransformer

    records = load_source(source)
    model = SentenceTransformer(str(embed_model))
    vectors = model.encode(
        [target_text(record) for record in records], normalize_embeddings=True, show_progress_bar=True
    )
    selected = greedy_kcenter(np.asarray(vectors), target)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{source}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for index in selected:
            handle.write(json.dumps(records[index], ensure_ascii=False) + "\n")
    report = {
        "source": source,
        "input_rows": len(records),
        "target": target,
        "selected_rows": len(selected),
        "embed_model": str(embed_model),
        "output": str(out),
    }
    (OUT_DIR / f"{source}.report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="source to compress")
    parser.add_argument("--target", type=int, default=None, help="target row count")
    parser.add_argument("--all", action="store_true", help="compress every over-represented source")
    parser.add_argument("--embed-model", type=Path, default=Path("models/bge-m3"))
    args = parser.parse_args()

    if args.all:
        for source, target in TARGETS.items():
            try:
                print(compress(source, target, args.embed_model), flush=True)
            except SystemExit as exc:
                print({"source": source, "skipped": str(exc)}, flush=True)
        return
    if not args.source:
        parser.error("give --source <name> [--target N] or --all")
    print(compress(args.source, args.target or TARGETS.get(args.source, 0), args.embed_model), flush=True)


if __name__ == "__main__":
    main()
