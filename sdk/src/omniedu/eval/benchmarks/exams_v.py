"""EXAMS-V: a transparent exact-label protocol over the released exam set.

Upstream publishes the data but no evaluation runner and no prompt, so this
adapter places the image and a short instruction that asks for the option label
alone, then parses that label.  ``--omit-image`` reproduces the text-only
ablation used for models without vision input; those runs are not comparable
with multimodal results, and the paper reports that the text-only protocol
degenerates to a constant answer on every model.

Data defaults to ``<data-root>/exams_v/data`` (parquet shards); the
decontaminated keep-set ships inside the package.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out, marker_failed
from ..io import resume
from ..metrics import build
from ..resources import clean_keep_ids

NAME = "EXAMS-V"
DEFAULT_MAX_TOKENS = 16
EXPECTED = 4797
PROMPT = (
    "Solve the multiple-choice exam question shown in the image. "
    "The question may be in any language. Reply with only the option label exactly as shown "
    "(for example A, B, C, D, E, a, b, c, d, \u0410, \u0411, \u0412, or \u0413). Do not provide an explanation."
)

#: option labels of the benchmark (Latin + Cyrillic)
LABELS = "ABCDE\u0410\u0411\u0412\u0413"
LABELS_WITH_DIGIT = LABELS + "2"
LETTER = r"0-9A-Za-z\u0410-\u042f\u0430-\u044f"

COLUMNS = [
    "image",
    "sample_id",
    "answer_key",
    "type",
    "grade",
    "subject",
    "subject_grouped",
    "language",
    "chemical_structure",
    "table",
    "figure",
    "graph",
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exams-data-dir", default=None, help="EXAMS-V parquet shards")
    parser.add_argument("--exams-keep-ids", default=None, help="decontaminated keep_ids.json")


def data_url(image: Any) -> str:
    raw = image.get("bytes") if isinstance(image, dict) else None
    if not raw and isinstance(image, dict) and image.get("path"):
        raw = Path(image["path"]).read_bytes()
    if not raw:
        raise ValueError("missing image bytes")
    mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def parse_answer(text: str, gold: Any) -> str:
    """Recover the option label from a response; ``''`` when nothing parses.

    A label only counts when it stands alone: prefer the label after the last
    answer marker, then a bracketed label, then the last standalone label (an
    answer sits at the end of a response).
    """
    stripped = (text or "").strip()
    compact = re.sub(r"[\s\[\](){}.:\uff1a*`]+", "", stripped)
    if compact.casefold() == str(gold).casefold():
        return str(gold)
    tail = None
    for match in re.finditer(r"(?:\u7b54\u6848|answer|option)", stripped, flags=re.IGNORECASE):
        tail = stripped[match.end() :]
    if tail:
        hit = re.search(
            r"[^%s]{0,8}([%s])(?![%s])" % (LETTER, LABELS_WITH_DIGIT, LETTER),
            tail,
            flags=re.IGNORECASE,
        )
        if hit:
            return hit.group(1)
    bracketed = re.findall(
        r"[(\uff08\[]\s*([%s])\s*[)\uff09\]]" % LABELS_WITH_DIGIT, stripped, flags=re.IGNORECASE
    )
    if bracketed:
        return bracketed[-1]
    standalone = re.findall(
        r"(?<![%s])([%s])(?![%s])" % (LETTER, LABELS, LETTER), stripped, flags=re.IGNORECASE
    )
    if standalone:
        return standalone[-1]
    return ""


def _payload(model: str, text: str, image: str | None, max_tokens: int) -> dict[str, Any]:
    if image is None:
        content: Any = text
    else:
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image}},
        ]
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    data_dir = Path(args.exams_data_dir or Path(args.data_root) / "exams_v" / "data")
    keep_ids = Path(args.exams_keep_ids or clean_keep_ids("exams_v"))
    keep = {str(entry["sample_id"]) for entry in json.loads(keep_ids.read_text(encoding="utf-8"))}
    if args.num_shards > 1:
        keep = {
            sample
            for sample in keep
            if zlib.crc32(sample.encode()) % args.num_shards == args.shard_index
        }
    total = len(keep)
    if args.num_shards == 1:
        assert total == EXPECTED, total

    output = Path(args.output_dir) / "predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = resume(output, key="sample_id", failed=lambda row: bool(row.get("error")))

    sink = Sink(output)
    max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS
    omit_image = bool(getattr(args, "omit_image", False))
    finished = 0

    async def worker(index: int, row: dict[str, Any]) -> None:
        nonlocal finished
        sample_id = str(row["sample_id"])
        if sample_id in done:
            return
        url = urls[index % len(urls)]
        error = None
        image = None
        if not omit_image:
            try:
                image = data_url(row["image"])
            except Exception as exc:
                image = ""
                error = repr(exc)
        response = ""
        if not error:
            response, _ = await chat(
                client, url, _payload(args.model, PROMPT, None if omit_image else image, max_tokens)
            )
        prediction = parse_answer(response, row["answer_key"])
        record = {
            "sample_id": sample_id,
            "model": args.model,
            "answer_key": str(row["answer_key"]),
            "prediction": prediction,
            "correct": prediction.casefold() == str(row["answer_key"]).casefold(),
            "response": response,
            "language": row["language"],
            "subject": row["subject"],
            "subject_grouped": row["subject_grouped"],
            "type": row["type"],
            "image_omitted": omit_image,
            "evaluation_protocol": "image_omitted_text_only" if omit_image else "image_supplied",
        }
        if error and not response:
            record["error"] = error
        await sink.write(record)
        finished += 1
        if finished % 100 == 0:
            print(f"[exams-v] {len(done) + finished}/{total}", flush=True)

    seen = 0
    async with httpx.AsyncClient(timeout=300) as client:
        for shard in sorted(data_dir.glob("test-*.parquet")):
            parquet = pq.ParquetFile(shard)
            rows = parquet.iter_batches(batch_size=max(8, args.concurrency), columns=COLUMNS)
            for batch in rows:
                chunk = [row for row in batch.to_pylist() if str(row["sample_id"]) in keep]
                seen += len(chunk)
                await fan_out(chunk, worker, concurrency=args.concurrency)
    if seen != total:
        print(
            f"[exams-v] data holds {seen} of {total} decontaminated ids (sampled or partial input)",
            flush=True,
        )

    return _score(output, args.model, omit_image)


def _rate(rows: list[dict[str, Any]]) -> float | None:
    return (sum(row["correct"] for row in rows) / len(rows) * 100) if rows else None


def _score(output: Path, model: str, omit_image: bool) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    unique = {row["sample_id"]: row for row in rows}
    values = list(unique.values())
    responses = [str(row.get("response") or "").strip() for row in values]
    uniq_resp = len(set(responses))
    top1_share = (max(Counter(responses).values()) / len(responses)) if responses else None
    correct = sum(row["correct"] for row in values)
    summary = {
        "benchmark": "EXAMS-V",
        "model": model,
        "count": len(values),
        "correct": correct,
        "accuracy": correct / len(values),
        "errors": sum(bool(row.get("error")) for row in values),
        "unparsed": sum(not str(row.get("prediction") or "") for row in values),
        "uniq_resp": uniq_resp,
        "top1_share": top1_share,
        "degenerate": uniq_resp == 1,
        "prompt": PROMPT,
        "image_omitted": omit_image,
        "evaluation_protocol": "image_omitted_text_only" if omit_image else "image_supplied",
        "note": (
            "Official repository releases data but no public evaluation runner/prompt; "
            "transparent exact-label adapter. Image-omitted runs are text-only ablations and "
            "are not directly comparable with multimodal results. uniq_resp==1 means every "
            "response was identical (degenerate) and the accuracy is not meaningful."
        ),
    }
    summary["metrics"] = build(
        "EXAMS-V",
        {
            "Overall": _rate(values),
            "Text": _rate([row for row in values if row.get("type") != "image_text"]),
            "Image+Text": _rate([row for row in values if row.get("type") == "image_text"]),
            "Natural Science": _rate(
                [r for r in values if r.get("subject_grouped") == "Natural Science"]
            ),
            "Social Sciences": _rate(
                [r for r in values if r.get("subject_grouped") == "Social Sciences"]
            ),
            "Other": _rate([r for r in values if r.get("subject_grouped") == "Other"]),
            "Chinese": _rate([row for row in values if row.get("language") == "Chinese"]),
            "English": _rate([row for row in values if row.get("language") == "English"]),
        },
    )
    return summary
