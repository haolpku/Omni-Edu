"""Run one judge prompt over one workload, with a strict output schema.

Every benchmark judge is required to answer with a single JSON object.  The
schema is selected from the prompt file name (see ``SCHEMAS``); a reply that does
not validate is retried with a corrective turn before it is written out as an
``error`` record.  Nothing is silently dropped:

* ``--base-url`` is normalised, so both ``http://host:3000`` and
  ``http://host:3000/v1`` are accepted;
* resuming skips only records that passed validation on an earlier pass;
* the process exits non-zero when anything is still failing, and prints an
  ``ok / failed`` summary.

MDK12-Bench and TutorBench are judged by ``judge_mdk12_tutorbench.py`` and
LongTutor's teaching quality by ``judge_longtutor.py``; each carries its own
rubric and validation, so there is deliberately no schema for them here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

from ..client import RateLimiter, normalize_base_url
from ..io import iter_jsonl, load_jsonl

REPAIR = (
    "Your previous reply could not be parsed. Reply again with ONLY one valid "
    "JSON object that matches the schema exactly - no markdown fence, no "
    "commentary before or after it."
)
TOTAL_SCORE_RE = re.compile(r"\u3010\u603b\u5206\u3011\s*(?:.*=)?\s*(\d+(?:\.\d*)?)\s*\u5206")


def as_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _yn(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "y"}
    return None


def _num(value: Any, lo: float | None = None, hi: float | None = None) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if lo is not None and number < lo:
        return None
    if hi is not None and number > hi:
        return None
    return number


def validate_gaokao(obj: dict[str, Any]) -> dict[str, Any] | None:
    score = _num(obj.get("score"), 0, None)
    if score is None:
        return None
    out: dict[str, Any] = {"score": score}
    maximum = _num(obj.get("max_score"), 0, None)
    if maximum is not None:
        if score > maximum:
            return None
        out["max_score"] = maximum
    reason = obj.get("reason")
    out["reason"] = reason if isinstance(reason, str) else ""
    return out


def validate_edumath(obj: dict[str, Any]) -> dict[str, Any] | None:
    keys = ["solvability", "solution_accuracy", "grade_appropriateness", "standards_alignment"]
    values = {}
    for key in keys:
        value = _yn(obj.get(key))
        if value is None:
            return None
        values[key] = value
    mac = obj.get("mac")
    expected = all(values.values())
    if not isinstance(mac, bool):
        mac = expected
    if mac != expected:
        return None
    reason = obj.get("reason")
    return {**values, "mac": mac, "reason": reason if isinstance(reason, str) else ""}


def validate_edumath_text(text: str) -> dict[str, Any] | None:
    """Official ``annotate.py`` rule for EDUMATH's LLM branch.

    Keep everything after the final ``<start_of_turn>model`` marker and take the
    FIRST ``Yes.`` / ``No.``; MaC is the share of ``Yes`` labels.
    """
    body = text.split("<start_of_turn>model")[-1] if "<start_of_turn>model" in text else text
    yes_index, no_index = body.find("Yes."), body.find("No.")
    if yes_index == -1 and no_index == -1:
        return None
    label = 1 if (yes_index != -1 and (no_index == -1 or yes_index < no_index)) else 0
    return {"label": label, "mac": bool(label), "reason": body.strip()[:2000]}


def validate_gaokao_text(text: str, maximum: float | None = None) -> dict[str, Any] | None:
    """GAOKAO-Bench official grader contract (the LAST ``【总分】 N 分``)."""
    matches = TOTAL_SCORE_RE.findall(text or "")
    if not matches:
        return None
    score = float(matches[-1])
    if maximum is not None and not (0 <= score <= float(maximum)):
        return None
    return {
        "score": score,
        "max_score": float(maximum) if maximum is not None else None,
        "occurrences": len(matches),
    }


SCHEMAS: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
    "gaokao_subjective_official": validate_gaokao,
    "edumath_official": validate_edumath,
}
TEXT_SCHEMAS: dict[str, Callable[..., dict[str, Any] | None]] = {
    "edumath_official": validate_edumath_text,
    "gaokao_subjective_official": validate_gaokao_text,
}


def render_user_template(template: str, record: dict[str, Any]) -> str:
    """Fill ``{field}`` placeholders from the record, leaving unknown ones."""
    flat = dict(record)
    meta = record.get("metadata")
    if isinstance(meta, dict):
        for key, value in meta.items():
            flat.setdefault(key, value)

    def substitute(match: re.Match[str]) -> str:
        value = flat.get(match.group(1))
        return str(value) if value is not None else match.group(0)

    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", substitute, template)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument(
        "--prompt-map",
        default=None,
        help="JSON {keyword: prompt_text}; the system prompt is selected per record",
    )
    parser.add_argument("--schema", default=None, help="schema name (required with --prompt-map)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="protocol budget; EDUMATH uses 2048 (omniedu.eval.judge.cli has the per-benchmark table)",
    )
    parser.add_argument("--format-retries", type=int, default=3)
    parser.add_argument(
        "--user-template",
        default=None,
        help="file whose {field} placeholders render the user message",
    )
    parser.add_argument("--rate-per-min", type=float, default=480.0)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--limit", type=int, help="judge only the first N records (smoke tests)")
    args = parser.parse_args(argv)

    if bool(args.prompt_file) == bool(args.prompt_map):
        print("give exactly one of --prompt-file / --prompt-map", file=sys.stderr)
        return 2
    schema_name = args.schema or (Path(args.prompt_file).stem if args.prompt_file else None)
    schema = SCHEMAS.get(schema_name)
    prompt_map = (
        json.loads(Path(args.prompt_map).read_text(encoding="utf-8")) if args.prompt_map else None
    )
    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    system = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None
    user_template = (
        Path(args.user_template).read_text(encoding="utf-8") if args.user_template else None
    )
    text_schema = TEXT_SCHEMAS.get(schema_name)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if output.exists():
        for record in iter_jsonl(output):
            if (
                record.get("judge_id")
                and record.get("error") is None
                and record.get("parsed") is not None
            ):
                done.add(record["judge_id"])
    todo = [row for row in rows if row.get("judge_id") not in done]
    label = (
        Path(args.prompt_file).stem
        if args.prompt_file
        else f"map:{Path(args.prompt_map).stem}" if args.prompt_map else "none"
    )
    print(
        f"[judge] schema={label} base={normalize_base_url(args.base_url)} "
        f"total={len(rows)} done={len(done)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        print(json.dumps({"ok": len(done), "failed": 0, "skipped": True}), flush=True)
        return 0

    client = AsyncOpenAI(
        base_url=normalize_base_url(args.base_url), api_key=args.api_key, timeout=300, max_retries=3
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    limiter = RateLimiter(args.rate_per_min)
    write_lock = asyncio.Lock()
    stats = {"ok": 0, "failed": 0}
    started = time.time()

    async def judge(record: dict[str, Any]) -> None:
        payload = {key: value for key, value in record.items() if key != "metadata"}
        meta = record.get("metadata")
        if isinstance(meta, dict):
            for key, value in meta.items():
                payload.setdefault(key, value)
        user_text = (
            render_user_template(user_template, record)
            if user_template is not None
            else json.dumps(payload, ensure_ascii=False)
        )
        if prompt_map is not None:
            keyword = record.get("keyword")
            if keyword is None and isinstance(meta, dict):
                keyword = meta.get("keyword")
            system_text = prompt_map.get(keyword) or prompt_map.get("__default__")
            if system_text is None:
                await _write(
                    output,
                    write_lock,
                    {
                        "judge_id": record.get("judge_id"),
                        "judge_model": args.model,
                        "judgment": "",
                        "parsed": None,
                        "error": f"no prompt for keyword={keyword!r} in --prompt-map",
                    },
                    stats,
                )
                return
        else:
            system_text = system

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        parsed = None
        text = ""
        error = None
        for attempt in range(1 + max(0, args.format_retries)):
            await limiter.wait()
            async with semaphore:
                try:
                    response = await client.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        temperature=0,
                        max_tokens=args.max_tokens,
                    )
                    text = response.choices[0].message.content or ""
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"[:400]
                    await asyncio.sleep(min(8, 2**attempt))
                    continue
            obj = as_json_obj(text)
            if obj is not None:
                parsed = schema(obj) if schema else obj
            if parsed is None and text_schema is not None:
                parsed = (
                    text_schema(text, record.get("score"))
                    if schema_name == "gaokao_subjective_official"
                    else text_schema(text)
                )
            if parsed is not None:
                error = None
                break
            error = "InvalidJudgeOutput: reply did not match the required JSON schema"
            messages = messages[:2] + [{"role": "user", "content": REPAIR}]
        if parsed is None and error is None:
            error = "InvalidJudgeOutput: empty reply"
        await _write(
            output,
            write_lock,
            {
                "judge_id": record.get("judge_id"),
                "judge_model": args.model,
                "judgment": text,
                "parsed": parsed,
                "error": error,
            },
            stats,
            started=started,
            total=len(todo),
        )

    await asyncio.gather(*(judge(row) for row in todo))
    print(
        json.dumps(
            {"ok": stats["ok"], "failed": stats["failed"], "output": str(output)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    if stats["failed"] and not args.allow_errors:
        return 1
    return 0


async def _write(
    output: Path,
    lock: asyncio.Lock,
    record: dict[str, Any],
    stats: dict[str, int],
    *,
    started: float | None = None,
    total: int = 0,
) -> None:
    async with lock:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats["ok" if record.get("parsed") is not None else "failed"] += 1
        count = sum(stats.values())
        if started is not None and count % 200 == 0:
            rate = count / max(1e-9, time.time() - started)
            print(
                f"[judge] {count}/{total} ok={stats['ok']} failed={stats['failed']} rate={rate:.1f}/s",
                flush=True,
            )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
