"""MDK12-Bench: official quick-inference generation plus deterministic scoring.

Generation uses the benchmark's own quick-inference system prompt and keeps the
benchmark images.  Objective rows (single choice, multiple choice, true/false)
are scored deterministically here; fill-in-the-blank and calculation rows are
left for the open-ended judge.

Defaults point at ``<official-root>/MDK12mini-*`` for the release and at the
packaged decontamination keep-set.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..bundle import read_jsonl, write_workload
from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out
from ..io import resume
from ..metrics import with_judged
from ..resources import clean_keep_ids

NAME = "MDK12-Bench"
DEFAULT_MAX_TOKENS = 4096
EXPECTED = 11951

# Byte-for-byte the official quick-inference system prompt: upstream writes the
# two ``\boxed`` occurrences with a doubled backslash, and the model sees exactly
# that.  Do not "tidy" it to a single backslash.
SYSTEM = (
    "Solve the question. The user asks a question, and you solves it. You first thinks about "
    "the reasoning process in the mind and then provides the user with the answer. The answer "
    "is in latex format and wrapped in $...$. The final answer must be wrapped using the "
    "\\\\boxed{} command. The reasoning process and answer are enclosed within <think> </think> "
    "and <answer> </answer> tags, respectively, i.e., <think> Since $1+1=2$, so the answer is "
    "$2$. </think><answer> The answer is $\\\\boxed{2}$ </answer>, which means assistant's output "
    "should start with <think> and end with </answer>."
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mdk12-keep-ids", default=None)


def load_rows(official_root: Path, keep_ids: Path) -> list[dict[str, Any]]:
    keep = {
        (entry["source_file"], str(entry["index"]))
        for entry in json.loads(keep_ids.read_text(encoding="utf-8"))
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(official_root.glob("MDK12mini-*/*/*.tsv")):
        relative = str(path.relative_to(official_root))
        frame = pd.read_csv(path, sep="\t")
        for record in frame.to_dict("records"):
            if (relative, str(record["index"])) in keep:
                record["_source_file"] = relative
                record["_sample_id"] = relative + "::" + str(record["index"])
                rows.append(record)
    if len(rows) != EXPECTED:
        print(
            f"[mdk12] loaded {len(rows)} rows (release has {EXPECTED}; sampled input?)", flush=True
        )
    return rows


def image_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = base64.b64decode(value)
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + value


def user_content(row: dict[str, Any], omit_image: bool) -> list[dict[str, Any]]:
    text = str(row["question"])
    options = []
    for letter in "ABCD":
        value = row.get(letter)
        if isinstance(value, str) and value.strip():
            options.append(f"{letter}. {value}")
    if options and not all(
        re.search(rf"(^|\n){letter}[.\uff0e\u3001:\uff1a)]", text)
        for letter in "ABCD"
        if any(item.startswith(letter + ".") for item in options)
    ):
        text += "\n" + "\n".join(options)
    parts: list[dict[str, Any]] = []
    url = image_url(row.get("image"))
    if url and not omit_image:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    parts.append({"type": "text", "text": text})
    return parts


def _payload(model: str, row: dict[str, Any], omit_image: bool, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content(row, omit_image)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    official = Path(getattr(args, "official", {}).get(NAME) or Path(args.data_root) / "mdk12_bench")
    keep_ids = Path(args.mdk12_keep_ids or clean_keep_ids("mdk12_bench"))
    rows = [
        row
        for index, row in enumerate(load_rows(official, keep_ids))
        if index % args.num_shards == args.shard_index
    ]

    output = Path(args.output_dir) / f"predictions.shard{args.shard_index}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = resume(output, key="_sample_id", failed=lambda row: bool(row.get("error")))
    todo = [row for row in rows if row["_sample_id"] not in done]
    print(
        f"[mdk12] shard {args.shard_index}: total={len(rows)} done={len(done)} todo={len(todo)}",
        flush=True,
    )

    sink = Sink(output, default=str)
    max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS
    omit_image = bool(getattr(args, "omit_image", False))
    completed = 0

    async def worker(index: int, row: dict[str, Any]) -> None:
        nonlocal completed
        url = urls[index % len(urls)]
        content, _ = await chat(client, url, _payload(args.model, row, omit_image, max_tokens))
        record = {key: value for key, value in row.items() if key != "image"}
        record.update(
            {
                "model": args.model,
                "prediction": content,
                "has_image": bool(row.get("image")),
                "image_omitted": bool(omit_image and row.get("image")),
                "evaluation_protocol": (
                    "image_omitted_text_only" if omit_image else "image_supplied"
                ),
            }
        )
        if content.startswith("__ERROR__"):
            record["error"] = content
        await sink.write(record)
        completed += 1
        if completed % 100 == 0:
            print(f"[mdk12] {len(done) + completed}/{len(rows)}", flush=True)

    async with httpx.AsyncClient(timeout=600) as client:
        await fan_out(
            todo, worker, concurrency=args.concurrency, chunk=max(64, args.concurrency * 4)
        )

    shards = sorted(output.parent.glob("predictions.shard*.jsonl")) or [output]
    summary = score(shards)
    write_workload(output.parent, "mdk12_open.jsonl", _open_workload(shards, args.model))
    return summary


def _open_workload(shards: list[Path], model: str) -> list[dict[str, Any]]:
    """Fill-in-the-blank and calculation rows, the part a judge has to score."""
    rows = []
    for shard in shards:
        for row in read_jsonl(shard):
            if row.get("category") not in ("fill_in_the_blank", "calculation"):
                continue
            rows.append(
                {
                    "judge_id": f"mdk12::{model}::{row.get('_sample_id')}",
                    "benchmark": "MDK12-Bench",
                    "sample_id": row.get("_sample_id"),
                    "question": row.get("question"),
                    "reference_answer": row.get("answer"),
                    "reference_solution": row.get("analysis"),
                    "model_response": row.get("prediction"),
                    "category": row.get("category"),
                    "knowledge": row.get("knowledge"),
                    "grade_level": row.get("grade_level"),
                    "difficulty_level": row.get("difficulty_level"),
                }
            )
    return rows


def _letters(text: str) -> set[str]:
    value = (text or "").upper()
    if "</THINK>" in value:
        value = value.split("</THINK>")[-1]
    boxes = re.findall(r"\\BOXED\{([^}]*)\}", value)
    value = boxes[-1] if boxes else value
    return set(re.findall(r"(?<![A-Z])[A-D](?![A-Z])", value))


def _truth(text: str) -> bool | None:
    value = (text or "").strip().lower()
    if any(marker in value for marker in ["\\boxed{true}", "\\boxed{t}", "\u6b63\u786e", " true"]):
        return True
    if any(
        marker in value for marker in ["\\boxed{false}", "\\boxed{f}", "\u9519\u8bef", " false"]
    ):
        return False
    return None


def _gold_truth(row: dict[str, Any]) -> bool | None:
    value = row.get("answer")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ["true", "t", "1", "yes", "\u6b63\u786e", "\u5bf9"]:
        return True
    if text in ["false", "f", "0", "no", "\u9519\u8bef", "\u9519"]:
        return False
    return None


def score(shards: list[Path]) -> dict[str, Any]:
    """Deterministic objective scoring of the generated prediction shards."""
    rows = [
        json.loads(line)
        for shard in shards
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = correct = 0
    by_category: dict[str, dict[str, Any]] = {}
    for row in rows:
        category = row["category"]
        ok: bool | None = None
        if category == "single_answer_choice":
            ok = _letters(row["prediction"]) == set(
                re.findall(r"[A-D]", str(row["answer"]).upper())
            )
        elif category == "multiple_answer_choice":
            ok = _letters(row["prediction"]) == set(
                re.findall(r"[A-D]", str(row["answer"]).upper())
            )
        elif category == "true_false":
            prediction = _truth(row["prediction"])
            gold = _gold_truth(row)
            ok = (prediction == gold) if prediction is not None and gold is not None else False
        if ok is not None:
            scored += 1
            correct += ok
            bucket = by_category.setdefault(category, {"count": 0, "correct": 0})
            bucket["count"] += 1
            bucket["correct"] += ok
    for bucket in by_category.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"] if bucket["count"] else None
    summary = {
        "benchmark": "MDK12mini",
        "records": len(rows),
        "objective_scored": scored,
        "objective_correct": correct,
        "objective_accuracy": correct / scored if scored else None,
        "errors": sum(bool(row.get("error")) for row in rows),
        "by_category": by_category,
        "judge_pending": sum(
            row["category"] in ["fill_in_the_blank", "calculation"] for row in rows
        ),
        "protocol": "official quick-inference SYSTEM_PROMPT; clean MDK12mini; deterministic objective scoring",
    }
    by_category = summary["by_category"]
    values: dict[str, float | None] = {
        "Single-choice": _percent(by_category.get("single_answer_choice", {}).get("accuracy")),
        "Multiple-choice": _percent(by_category.get("multiple_answer_choice", {}).get("accuracy")),
        "True/False": _percent(by_category.get("true_false", {}).get("accuracy")),
        "Objective Avg": _percent(summary["objective_accuracy"]),
    }
    summary["metrics"] = with_judged("MDK12-Bench", shards[0].parent, values)
    if values.get("Open Correctness") is not None and summary["objective_accuracy"] is not None:
        objective = summary["objective_scored"] or 0
        open_count = summary["judge_pending"] or 0
        total = objective + open_count
        if total:
            summary["metrics"]["Full Overall"] = (
                (
                    summary["objective_accuracy"] * objective
                    + values["Open Correctness"] / 100 * open_count
                )
                / total
                * 100
            )
    return summary


def _percent(value: Any) -> float | None:
    return None if value is None else float(value) * 100
