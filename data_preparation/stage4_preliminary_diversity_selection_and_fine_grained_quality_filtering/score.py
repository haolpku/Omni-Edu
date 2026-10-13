#!/usr/bin/env python3
"""Fine-grained quality scoring with the task rubrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import llm

import rubrics

HERE = Path(__file__).resolve().parent
POOLS = HERE / "pools"
RESULTS = HERE / "results"


def family_of(record: dict) -> str:
    kind = str(record.get("kind") or "")
    if record.get("essay"):
        return "sr_2_writing_feedback"
    if kind == "reading":
        return "sr_1_problem_solving"
    if kind == "longitudinal_tutoring":
        return "sr_6_longitudinal_tutoring"
    if kind == "mathtutor_scaffolding":
        return "sr_8_minimal_scaffold"
    if kind == "tutorbench_dialogue":
        return "sr_9_multi_turn_dialogue"
    if kind == "oatutor_guidance":
        return "sr_10_hint_based_guidance"
    if kind == "strict_alignment_binary":
        return "sr_4_strict_standard_alignment"
    if record.get("category") == "curriculum_grounding":
        return "sr_3_curriculum_label"
    if record.get("category") == "diagnostic_reasoning":
        return "sr_5_error_diagnosis"
    if record.get("category") == "pedagogical_action":
        return "sr_7_general_pedagogical_action"
    return "sr_1_problem_solving"


def render(record: dict) -> str:
    keys = (
        "passage", "prompt", "problem", "question", "options", "answer", "answer_text", "cot",
        "essay", "scores", "feedback", "messages", "conversation", "hint", "student_answer",
        "history", "annotation", "image_ref", "images", "context_messages", "candidate_teacher_turn",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default=None, help="score one capability only")
    parser.add_argument("--limit", type=int, default=0)
    llm.add_common_args(parser)
    args = parser.parse_args()

    records: list[dict] = []
    for pool in sorted(POOLS.glob("*.jsonl")):
        if args.category and pool.stem != args.category:
            continue
        with pool.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no candidate pool in {POOLS}; run pools.py first")
    if args.limit:
        records = records[: args.limit]

    out = RESULTS / "scores.jsonl"
    done = llm.load_done(out, "record_id", lambda row: bool(row.get("scores")))
    todo = []
    for index, record in enumerate(records):
        record["record_id"] = f"{record.get('category')}::{record.get('source')}::{record.get('uid', index)}"
        record["family"] = family_of(record)
        if record["record_id"] not in done:
            todo.append(record)
    print(f"todo={len(todo)} skip={len(done)} out={out}", flush=True)

    asyncio.run(
        llm.run(
            todo,
            out,
            lambda record: (rubrics.system_prompt(record["family"], record), render(record)),
            lambda record, obj, error: {
                "record_id": record["record_id"],
                "dataset": record.get("source"),
                "category": record.get("category"),
                "family": record["family"],
                "scores": rubrics.numeric_scores(obj or {}),
                "reason": str((obj or {}).get("reason", ""))[:300],
                "error": error,
            },
            key="record_id",
            chat=llm.from_args(args),
            concurrency=args.concurrency,
        )
    )
    print({"output": str(out), "scored": len(todo)}, flush=True)


if __name__ == "__main__":
    main()
