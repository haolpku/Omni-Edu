"""Validated judge for MDK12-Bench open-ended and TutorBench rubric scoring.

This is the judge that produced the paper's MDK12 open-ended and TutorBench
numbers, so it is the definition of those columns.  Each task has an explicit
JSON contract; a reply is persisted as successful only after task-specific
validation, and malformed replies are retried in place.

The prompt text below is the paper's, unchanged.  Do NOT replace it with a
pointer at some upstream rubric: both benchmarks publish data but no reference
judge, so this contract *is* the rubric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from ..client import RateLimiter, normalize_base_url
from ..io import iter_jsonl, load_jsonl

TASKS = {
    "mdk12": ("MDK12-Bench", "open_qa_calculation", 1024),
    "tutorbench": ("TutorBench", "rubric_criteria", 6144),
}


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    return value


def mdk_prompt(row: dict[str, Any]) -> tuple[str, str, int]:
    system = (
        "You are grading a K-12 open-answer math question. Compare the student's answer with the "
        "reference answer and reference solution. Judge mathematical correctness, not wording. "
        "Return JSON only with exactly this schema: "
        '{"score": 0.0, "reason": "brief justification"}. '
        "score must be a number from 0.0 to 1.0, where 1 is fully correct, 0 is fully incorrect, "
        "and partial credit is allowed. Do not add markdown or other keys."
    )
    payload = {
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "reference_solution": row.get("reference_solution"),
        "student_answer": row.get("model_response"),
    }
    return system, json.dumps(payload, ensure_ascii=False), 0


def tutor_prompt(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    rubrics = row.get("rubrics") or []
    if isinstance(rubrics, str):
        rubrics = json.loads(rubrics)
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError("TutorBench row has no valid rubric list")
    rubric_ids = [f"R{index:02d}" for index in range(1, len(rubrics) + 1)]
    system = (
        "You are grading a tutor response against independent rubric criteria. Evaluate every rubric ID "
        "independently. Return only one flat JSON object mapping every supplied rubric ID to integer 0 or 1. "
        "Use 1 when the tutor response passes that rubric and 0 otherwise. The output keys must match the "
        "required_output_template exactly: do not omit, merge, rename, reorder, or add keys. Do not provide "
        "reasons, markdown, or any text outside the JSON object."
    )
    payload = {
        "question": row.get("question"),
        "previous_tutor_explanation": row.get("previous_tutor_explanation"),
        "student_follow_up": row.get("student_follow_up"),
        "tutor_response": row.get("model_response"),
        "rubrics": [
            {
                "id": rubric_id,
                "criteria": rubric.get("criteria"),
                "attributes": rubric.get("attributes"),
            }
            for rubric_id, rubric in zip(rubric_ids, rubrics)
        ],
        "required_output_template": {rubric_id: 0 for rubric_id in rubric_ids},
    }
    return system, json.dumps(payload, ensure_ascii=False), rubric_ids


def validate(task: str, obj: dict[str, Any], expected: Any) -> dict[str, Any]:
    if task == "mdk12":
        if set(obj) != {"score", "reason"}:
            raise ValueError(f"MDK12 keys must be score/reason, got {sorted(obj)}")
        score = obj["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= float(score) <= 1
        ):
            raise ValueError("MDK12 score must be numeric in [0,1]")
        if not isinstance(obj["reason"], str) or not obj["reason"].strip():
            raise ValueError("MDK12 reason must be non-empty")
        return {"score": float(score), "reason": obj["reason"].strip()}
    expected_ids = list(expected)
    if list(obj) != expected_ids:
        missing = [key for key in expected_ids if key not in obj]
        extra = [key for key in obj if key not in expected_ids]
        raise ValueError(f"TutorBench ID map mismatch: missing={missing}, extra={extra}")
    normalized = []
    for position, rubric_id in enumerate(expected_ids, 1):
        score = obj[rubric_id]
        if isinstance(score, bool) or score not in (0, 1):
            raise ValueError(f"invalid score for {rubric_id}")
        normalized.append(
            {
                "index": position,
                "rubric_id": rubric_id,
                "pass": bool(score),
                "score": int(score),
            }
        )
    return {"criteria": normalized, "protocol": "fixed_rubric_id_map_v1"}


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-tag", default="unknown")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--rate-per-min", type=float, default=420.0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args(argv)

    benchmark, manifest_task, default_max_tokens = TASKS[args.task]
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    build = mdk_prompt if args.task == "mdk12" else tutor_prompt
    # Validate every input before making any API request.
    expected = [build(row)[2] for row in rows]
    sizes = [len(value) if isinstance(value, list) else int(value) for value in expected]
    print(
        json.dumps(
            {
                "benchmark": benchmark,
                "model": args.model_tag,
                "rows": len(rows),
                "min_expected_items": min(sizes) if sizes else 0,
                "max_expected_items": max(sizes) if sizes else 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    max_tokens = args.max_tokens or default_max_tokens
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {
        record["judge_id"]
        for record in (iter_jsonl(output) if output.exists() else ())
        if record.get("judge_id")
        and record.get("error") is None
        and record.get("parsed") is not None
    }
    todo = [row for row in rows if row["judge_id"] not in done]
    print(json.dumps({"already_done": len(done), "todo": len(todo)}), flush=True)
    if not todo:
        return 0

    client = AsyncOpenAI(
        base_url=normalize_base_url(args.base_url), api_key=args.api_key, timeout=600, max_retries=0
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    limiter = RateLimiter(args.rate_per_min)
    lock = asyncio.Lock()
    stats = {"new": 0, "errors": 0}

    async def judge(row: dict[str, Any]) -> None:
        system, user, expected_items = build(row)
        response_text = ""
        parsed = None
        last_error = None
        for attempt in range(args.max_retries):
            await limiter.wait()
            try:
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=args.judge_model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                response_text = response.choices[0].message.content or ""
                parsed = validate(args.task, extract_json(response_text), expected_items)
                last_error = None
                break
            except Exception as exc:
                last_error = repr(exc)
                if attempt + 1 < args.max_retries:
                    await asyncio.sleep(min(2**attempt, 16))
        record = {
            "judge_id": row["judge_id"],
            "benchmark": benchmark,
            "task": manifest_task,
            "evaluated_model": row.get("model", args.model_tag),
            "judge_model": args.judge_model,
            "parsed": parsed,
            "judgment": response_text,
            "error": last_error,
        }
        async with lock:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["new"] += 1
            stats["errors"] += bool(last_error)
            if stats["new"] % 50 == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)

    await asyncio.gather(*(judge(row) for row in todo))
    await client.close()
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if stats["errors"] and not args.allow_errors:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
