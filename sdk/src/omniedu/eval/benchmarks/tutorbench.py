"""TutorBench: one tutor response per task, using the benchmark's own prompt.

Generation replays the benchmark prompt, rubric metadata and (where present) the
student's uploaded work; the images are dropped under ``--omit-image`` for
text-only models.  Scoring is the rubric judge in ``omniedu.eval.judge``.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..bundle import read_jsonl, write_workload
from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out
from ..io import resume
from ..metrics import with_judged
from ..resources import clean_keep_ids

NAME = "TutorBench"
DEFAULT_MAX_TOKENS = 1024

SYSTEM = (
    "You are a highly capable, supportive tutor. Respond directly to the student. Be accurate, "
    "pedagogically helpful, appropriately calibrated to the student, and avoid unnecessary verbosity."
)
COLUMNS = [
    "TASK_ID",
    "BATCH",
    "SUBJECT",
    "PROMPT",
    "IMAGE_URL",
    "UC1_INITIAL_EXPLANATION",
    "FOLLOW_UP_PROMPT",
    "RUBRICS",
    "bloom_taxonomy",
    "Image",
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tutorbench-data-dir", default=None)
    parser.add_argument("--tutorbench-keep-ids", default=None)


def image_url(image: Any) -> str | None:
    raw = image.get("bytes") if isinstance(image, dict) else None
    if not raw and isinstance(image, dict) and image.get("path"):
        raw = Path(image["path"]).read_bytes()
    if not raw:
        return None
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(raw).decode()


def messages(row: dict[str, Any], omit_image: bool) -> list[dict[str, Any]]:
    batch = str(row.get("BATCH", ""))
    prompt = str(row.get("PROMPT") or "").strip()
    initial = str(row.get("UC1_INITIAL_EXPLANATION") or "").strip()
    follow = str(row.get("FOLLOW_UP_PROMPT") or "").strip()
    if batch.startswith("USE_CASE_1") and initial and follow:
        result = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial},
        ]
        user_text = follow
    else:
        parts = [prompt]
        if initial:
            parts.append("Previous tutor explanation:\n" + initial)
        if follow:
            parts.append("Student work or follow-up:\n" + follow)
        user_text = "\n\n".join(part for part in parts if part)
        result = [{"role": "system", "content": SYSTEM}]
    url = image_url(row.get("Image"))
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    if url and not omit_image:
        content.append({"type": "image_url", "image_url": {"url": url}})
    result.append({"role": "user", "content": content})
    return result


def _payload(model: str, row: dict[str, Any], omit_image: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages(row, omit_image),
        "temperature": 0,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    data_dir = Path(args.tutorbench_data_dir or Path(args.data_root) / "tutorbench")
    keep_ids = Path(args.tutorbench_keep_ids or clean_keep_ids("tutorbench"))
    keep = {str(entry["TASK_ID"]) for entry in json.loads(keep_ids.read_text(encoding="utf-8"))}

    output = Path(args.output_dir) / f"predictions.shard{args.shard_index}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = resume(output, key="TASK_ID", failed=lambda row: bool(row.get("error")))

    sink = Sink(output)
    omit_image = bool(getattr(args, "omit_image", False))
    assigned = 0
    completed = 0
    global_index = 0
    sources: dict[str, dict[str, Any]] = {}

    async def worker(index: int, row: dict[str, Any]) -> None:
        nonlocal completed
        if str(row["TASK_ID"]) in done:
            return
        url = urls[index % len(urls)]
        content, _ = await chat(client, url, _payload(args.model, row, omit_image))
        record = {
            "TASK_ID": str(row["TASK_ID"]),
            "model": args.model,
            "batch": row.get("BATCH"),
            "subject": row.get("SUBJECT"),
            "response": content,
            "rubrics": row.get("RUBRICS"),
            "bloom_taxonomy": row.get("bloom_taxonomy"),
            "has_image": bool(row.get("Image")),
            "image_omitted": bool(omit_image and row.get("Image")),
            "evaluation_protocol": "image_omitted_text_only" if omit_image else "image_supplied",
            "error": content if content.startswith("__ERROR__") else None,
        }
        await sink.write(record)
        completed += 1
        if completed % 50 == 0:
            print(f"[tutorbench] {completed}", flush=True)

    async with httpx.AsyncClient(timeout=600) as client:
        for path in sorted(data_dir.glob("*.parquet")):
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=max(8, args.concurrency), columns=COLUMNS):
                chunk = []
                for row in batch.to_pylist():
                    if str(row["TASK_ID"]) in keep:
                        sources[str(row["TASK_ID"])] = row
                        if global_index % args.num_shards == args.shard_index:
                            chunk.append(row)
                        global_index += 1
                assigned += len(chunk)
                if chunk:
                    await fan_out(chunk, worker, concurrency=args.concurrency)

    shards = sorted(output.parent.glob("predictions.shard*.jsonl")) or [output]
    return _score(shards, sources, args.model, args.shard_index, assigned, completed)


def _score(
    shards: list[Path],
    sources: dict[str, dict[str, Any]],
    model: str,
    shard_index: int,
    assigned: int,
    new: int,
) -> dict[str, Any]:
    rows = [row for shard in shards for row in read_jsonl(shard)]
    write_workload(
        shards[0].parent,
        "tutorbench_rubric.jsonl",
        [_rubric_workload(row, sources.get(row["TASK_ID"], {}), model) for row in rows],
    )
    return {
        "benchmark": "TutorBench",
        "model": model,
        "shard": shard_index,
        "count": len(rows),
        "assigned": assigned,
        "new": new,
        "errors": sum(bool(row.get("error")) for row in rows),
        "note": "rubric scoring is a separate judge step (omniedu.eval.judge)",
        "metrics": with_judged("TutorBench", shards[0].parent, {}),
    }


def _rubric_workload(
    prediction: dict[str, Any], source: dict[str, Any], model: str
) -> dict[str, Any]:
    task_id = str(prediction["TASK_ID"])
    return {
        "judge_id": f"tutorbench::{model}::{task_id}",
        "benchmark": "TutorBench",
        "model": model,
        "sample_id": task_id,
        "question": source.get("PROMPT"),
        "previous_tutor_explanation": source.get("UC1_INITIAL_EXPLANATION"),
        "student_follow_up": source.get("FOLLOW_UP_PROMPT"),
        "model_response": prediction.get("response"),
        "rubrics": source.get("RUBRICS") or prediction.get("rubrics"),
        "subject": source.get("SUBJECT") or prediction.get("subject"),
        "batch": source.get("BATCH") or prediction.get("batch"),
        "bloom_taxonomy": source.get("bloom_taxonomy") or prediction.get("bloom_taxonomy"),
        "has_image": prediction.get("has_image"),
    }
