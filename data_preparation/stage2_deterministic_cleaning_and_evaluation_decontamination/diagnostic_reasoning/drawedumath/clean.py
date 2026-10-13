#!/usr/bin/env python3
"""Deterministic fast-clean and exact-collision audit for DrawEduMath.

Teacher-authored QA is the only training gold emitted to records.jsonl.
GPT-4o/Claude facets and QA are retained separately in provenance.jsonl.
DrawEduMath_Model_Predictions.csv is inventoried but never read as gold.
No source annotation is rewritten and no model is called.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools
from typing import Any, Iterable

from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = pools.source_dir("diagnostic_reasoning")
DEFAULT_DATA = SOURCE_ROOT / "huggingface" / "DrawEduMath" / "Data"
DEFAULT_SCRATCH = pools.cleaned_dir("diagnostic_reasoning", "ScratchMath")
DEFAULT_ERROR = pools.cleaned_dir("diagnostic_reasoning", "ErrorRadar")
SOURCE_COLUMNS = [
    "Problem ID",
    "Image Name",
    "Image URL",
    "Image SHA256",
    "Image Caption",
    "Facets By GPT4o",
    "Facets By Claude",
    "QA Teacher",
    "QA GPT4o",
    "QA Claude",
]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
WS = re.compile(r"\s+")
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_PIXELS = 100_000_000


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def collision_text(value: Any) -> str:
    return WS.sub(" ", normalize_text(value))


def parse_json_list(raw: str, field: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json:{field}:{exc.msg}") from exc
    if not isinstance(value, list):
        raise ValueError(f"not_a_list:{field}")
    return value


def normalize_string_list(raw: str, field: str) -> list[str]:
    values = parse_json_list(raw, field)
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise ValueError(f"invalid_string_item:{field}:{index}")
        result.append(normalize_text(value))
    return result


def normalize_qa_list(raw: str, field: str) -> list[dict[str, str]]:
    values = parse_json_list(raw, field)
    result = []
    for index, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"question", "answer"}:
            raise ValueError(f"invalid_qa_shape:{field}:{index}")
        question = normalize_text(value["question"])
        answer = normalize_text(value["answer"])
        if not question or not answer:
            raise ValueError(f"blank_qa_field:{field}:{index}")
        result.append({"question": question, "answer": answer})
    return result


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(data: bytes, allowed_formats: set[str] | None = None) -> dict[str, Any]:
    if not data:
        raise ValueError("empty_image")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            media_type = Image.MIME.get(image_format)
    except Exception as exc:
        raise ValueError(f"invalid_image:{type(exc).__name__}") from exc
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise ValueError("unsafe_image_dimensions")
    allowed_formats = allowed_formats or {"JPEG", "PNG"}
    if image_format not in allowed_formats or not media_type:
        raise ValueError(f"unsupported_image_format:{image_format}")
    return {
        "bytes": len(data),
        "width": width,
        "height": height,
        "format": image_format,
        "media_type": media_type,
    }


def download_bytes(url: str, timeout: int = 60, attempts: int = 4) -> bytes:
    parts = urllib.parse.urlsplit(url)
    quoted_url = urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc.encode("idna").decode("ascii"),
            urllib.parse.quote(urllib.parse.unquote(parts.path), safe="/:@"),
            urllib.parse.quote(urllib.parse.unquote(parts.query), safe="=&/:@"),
            "",
        )
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                quoted_url, headers={"User-Agent": "DrawEduMath-fast-clean/1.0"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(MAX_IMAGE_BYTES + 1)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def verify_and_store_image(item: tuple[int, str, str, str], output_dir: Path) -> dict[str, Any]:
    source_row, image_name, url, expected_sha = item
    suffix = Path(image_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".bin"
    relative = Path("images") / expected_sha[:2] / f"{expected_sha}{suffix}"
    destination = output_dir / relative
    try:
        if destination.is_file() and sha256_file(destination) == expected_sha:
            data = destination.read_bytes()
            source = "existing_cache"
        else:
            data = download_bytes(url)
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"sha256_mismatch:{actual_sha}")
            metadata = validate_image(data)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            temporary.write_bytes(data)
            os.replace(temporary, destination)
            source = "downloaded"
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"sha256_mismatch:{actual_sha}")
        metadata = validate_image(data)
        return {
            "source_row_index": source_row,
            "image_name": image_name,
            "url": url,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "status": "verified",
            "storage_source": source,
            "path": relative.as_posix(),
            **metadata,
        }
    except Exception as exc:
        return {
            "source_row_index": source_row,
            "image_name": image_name,
            "url": url,
            "expected_sha256": expected_sha,
            "status": "failed",
            "reason": str(exc),
        }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_cross_indexes(scratch_dir: Path, error_dir: Path) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, int],
]:
    image_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    question_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scratch_rows = list(read_jsonl(scratch_dir / "kept.jsonl"))
    error_rows = list(read_jsonl(error_dir / "kept.jsonl"))
    for row in scratch_rows:
        image = row.get("student_scratchwork") or {}
        sha = image.get("sha256")
        if sha:
            image_index[sha].append(
                {
                    "dataset": "ScratchMath",
                    "question_id": row.get("question_id"),
                    "subset": row.get("subset"),
                }
            )
        key = collision_text(row.get("question"))
        if key:
            question_index[key].append(
                {
                    "dataset": "ScratchMath",
                    "question_id": row.get("question_id"),
                    "subset": row.get("subset"),
                }
            )
    for row in error_rows:
        key = collision_text(row.get("problem"))
        if key:
            question_index[key].append(
                {
                    "dataset": "ErrorRadar",
                    "source_record_id": row.get("source_record_id"),
                }
            )
    return (
        image_index,
        question_index,
        {
            "scratchmath_records": len(scratch_rows),
            "scratchmath_unique_image_sha256": len(image_index),
            "errorradar_records": len(error_rows),
        },
    )


def audit_errorradar_images(error_dir: Path, workers: int) -> dict[str, Any]:
    rows = list(read_jsonl(error_dir / "kept.jsonl"))
    unique_urls = sorted(
        {normalize_text(row.get("content_image")) for row in rows if row.get("content_image")}
    )

    def fetch(url: str) -> dict[str, Any]:
        try:
            data = download_bytes(url)
            metadata = validate_image(
                data, allowed_formats={"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}
            )
            return {
                "url": url,
                "status": "verified",
                "sha256": hashlib.sha256(data).hexdigest(),
                **metadata,
            }
        except Exception as exc:
            return {"url": url, "status": "failed", "reason": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(fetch, unique_urls))
    return {
        "source": str(error_dir / "kept.jsonl"),
        "unique_urls": len(unique_urls),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=pools.cleaned_dir("diagnostic_reasoning", "DrawEduMath"))
    parser.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument("--error-dir", type=Path, default=DEFAULT_ERROR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--skip-errorradar-image-download",
        action="store_true",
        help="Still audits exact questions and ScratchMath SHA collisions.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_csv = data_dir / "DrawEduMath_QA.csv"
    predictions_csv = data_dir / "DrawEduMath_Model_Predictions.csv"

    raw_rows: list[dict[str, str]] = []
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SOURCE_COLUMNS:
            raise ValueError(f"source schema mismatch: {reader.fieldnames}")
        raw_rows = list(reader)

    image_jobs: list[tuple[int, str, str, str]] = []
    parsed_images: list[dict[str, Any]] = []
    rejected_images: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_rows):
        try:
            problem_id = normalize_text(raw["Problem ID"])
            image_name = normalize_text(raw["Image Name"])
            image_url = normalize_text(raw["Image URL"])
            image_sha = normalize_text(raw["Image SHA256"]).lower()
            caption = normalize_text(raw["Image Caption"])
            if not all((problem_id, image_name, image_url, caption)):
                raise ValueError("blank_required_image_field")
            if not HEX64.fullmatch(image_sha):
                raise ValueError("invalid_image_sha256")
            teacher_qa = normalize_qa_list(raw["QA Teacher"], "QA Teacher")
            if not teacher_qa:
                raise ValueError("empty_teacher_qa")
            parsed_images.append(
                {
                    "source_row_index": row_index,
                    "problem_id": problem_id,
                    "image_name": image_name,
                    "image_url": image_url,
                    "image_sha256": image_sha,
                    "image_caption": caption,
                    "teacher_qa": teacher_qa,
                    "facets_gpt4o": normalize_string_list(
                        raw["Facets By GPT4o"], "Facets By GPT4o"
                    ),
                    "facets_claude": normalize_string_list(
                        raw["Facets By Claude"], "Facets By Claude"
                    ),
                    "qa_gpt4o": normalize_qa_list(raw["QA GPT4o"], "QA GPT4o"),
                    "qa_claude": normalize_qa_list(raw["QA Claude"], "QA Claude"),
                }
            )
            image_jobs.append((row_index, image_name, image_url, image_sha))
        except ValueError as exc:
            rejected_images.append({"source_row_index": row_index, "reason": str(exc)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        image_checks = list(
            pool.map(lambda item: verify_and_store_image(item, output_dir), image_jobs)
        )
    image_checks.sort(key=lambda row: row["source_row_index"])
    image_check_by_row = {row["source_row_index"]: row for row in image_checks}
    verified_images = [
        row
        for row in parsed_images
        if image_check_by_row[row["source_row_index"]]["status"] == "verified"
    ]
    for row in image_checks:
        if row["status"] != "verified":
            rejected_images.append(
                {
                    "source_row_index": row["source_row_index"],
                    "reason": "image_reference_verification_failed",
                    "detail": row.get("reason"),
                }
            )

    image_index, question_index, cross_source_counts = load_cross_indexes(
        args.scratch_dir.resolve(), args.error_dir.resolve()
    )
    records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    seen_records: dict[str, str] = {}
    seen_image_rows: dict[str, str] = {}
    duplicate_image_sha_rows = 0

    for image_row in verified_images:
        check = image_check_by_row[image_row["source_row_index"]]
        image_record_id = hashlib.sha256(
            canonical_json(
                {
                    "problem_id": image_row["problem_id"],
                    "image_sha256": image_row["image_sha256"],
                }
            ).encode()
        ).hexdigest()[:24]
        image_key = canonical_json(
            {
                key: image_row[key]
                for key in (
                    "problem_id",
                    "image_sha256",
                    "image_caption",
                    "teacher_qa",
                )
            }
        )
        if image_key in seen_image_rows:
            duplicate_image_sha_rows += 1
        else:
            seen_image_rows[image_key] = image_record_id
        provenance.append(
            {
                "dataset": "DrawEduMath",
                "image_record_id": image_record_id,
                "problem_id": image_row["problem_id"],
                "image_sha256": image_row["image_sha256"],
                "synthetic_annotations_not_gold": {
                    "facets_gpt4o": image_row["facets_gpt4o"],
                    "facets_claude": image_row["facets_claude"],
                    "qa_gpt4o": image_row["qa_gpt4o"],
                    "qa_claude": image_row["qa_claude"],
                },
            }
        )
        if image_row["image_sha256"] in image_index:
            collisions.append(
                {
                    "collision_type": "image_sha256",
                    "drawedumath_image_record_id": image_record_id,
                    "drawedumath_image_sha256": image_row["image_sha256"],
                    "matches": image_index[image_row["image_sha256"]],
                }
            )
        for qa_index, qa in enumerate(image_row["teacher_qa"]):
            record = {
                "dataset": "DrawEduMath",
                "split": "unsplit",
                "gold_source": "teacher",
                "problem_id": image_row["problem_id"],
                "image_record_id": image_record_id,
                "image": {
                    "name": image_row["image_name"],
                    "url": image_row["image_url"],
                    "sha256": image_row["image_sha256"],
                    "path": check["path"],
                    "media_type": check["media_type"],
                    "bytes": check["bytes"],
                    "width": check["width"],
                    "height": check["height"],
                },
                "image_caption": image_row["image_caption"],
                "question": qa["question"],
                "answer": qa["answer"],
                "source_qa_index": qa_index,
            }
            dedup_payload = {
                key: record[key]
                for key in (
                    "problem_id",
                    "image_record_id",
                    "image_caption",
                    "question",
                    "answer",
                )
            }
            key = canonical_json(dedup_payload)
            record_id = hashlib.sha256(key.encode()).hexdigest()[:24]
            record["record_id"] = record_id
            if key in seen_records:
                rejected_records.append(
                    {
                        "reason": "exact_normalized_duplicate",
                        "record_id": record_id,
                        "duplicate_of": seen_records[key],
                        "source_row_index": image_row["source_row_index"],
                        "source_qa_index": qa_index,
                    }
                )
                continue
            seen_records[key] = record_id
            records.append(record)
            question_key = collision_text(qa["question"])
            if question_key in question_index:
                collisions.append(
                    {
                        "collision_type": "teacher_question_exact_normalized",
                        "drawedumath_record_id": record_id,
                        "normalized_text": question_key,
                        "matches": question_index[question_key],
                    }
                )

    error_image_audit: dict[str, Any] | None = None
    error_image_sha_index: dict[str, list[str]] = defaultdict(list)
    if not args.skip_errorradar_image_download:
        error_image_audit = audit_errorradar_images(args.error_dir.resolve(), args.workers)
        for row in error_image_audit["results"]:
            if row["status"] == "verified":
                error_image_sha_index[row["sha256"]].append(row["url"])
        for image_row in verified_images:
            matches = error_image_sha_index.get(image_row["image_sha256"])
            if matches:
                collisions.append(
                    {
                        "collision_type": "image_sha256",
                        "drawedumath_image_sha256": image_row["image_sha256"],
                        "matches": [
                            {"dataset": "ErrorRadar", "content_image": url} for url in matches
                        ],
                    }
                )

    records.sort(key=lambda row: (row["source_qa_index"], row["record_id"]))
    provenance.sort(key=lambda row: row["image_record_id"])
    collisions.sort(key=canonical_json)
    rejected_images.sort(key=lambda row: row["source_row_index"])
    write_jsonl(output_dir / "kept.jsonl", records)
    write_jsonl(output_dir / "provenance.jsonl", provenance)
    write_jsonl(output_dir / "image_manifest.jsonl", image_checks)
    write_jsonl(output_dir / "rejected.jsonl", rejected_images + rejected_records)
    write_jsonl(output_dir / "cross_dataset_collisions.jsonl", collisions)
    if error_image_audit is not None:
        write_jsonl(
            output_dir / "errorradar_image_audit.jsonl",
            error_image_audit["results"],
        )

    teacher_qa_total = sum(len(row["teacher_qa"]) for row in verified_images)
    synthetic_counts = {
        "facets_gpt4o": sum(len(row["facets_gpt4o"]) for row in verified_images),
        "facets_claude": sum(len(row["facets_claude"]) for row in verified_images),
        "qa_gpt4o": sum(len(row["qa_gpt4o"]) for row in verified_images),
        "qa_claude": sum(len(row["qa_claude"]) for row in verified_images),
    }
    rejection_counts = Counter(row["reason"] for row in rejected_images + rejected_records)
    collision_counts = Counter(row["collision_type"] for row in collisions)
    unique_image_shas = {row["image_sha256"] for row in verified_images}
    prediction_inventory = {
        "path": str(predictions_csv),
        "exists": predictions_csv.is_file(),
        "bytes": predictions_csv.stat().st_size if predictions_csv.is_file() else None,
        "policy": "excluded_entirely_from_training_gold; file not parsed by this pipeline",
    }
    report = {
        "dataset": "DrawEduMath",
        "scope": "deterministic_clean_no_rewrite_no_model",
        "source_file": str(source_csv),
        "source_rows": len(raw_rows),
        "source_schema_exact": True,
        "image_rows": {
            "parsed": len(parsed_images),
            "verified_and_kept": len(verified_images),
            "rejected": len(rejected_images),
            "unique_sha256": len(unique_image_shas),
            "exact_duplicate_image_annotations": duplicate_image_sha_rows,
        },
        "gold_teacher_qa": {
            "source_pairs_after_image_validation": teacher_qa_total,
            "kept_records": len(records),
            "exact_duplicates_removed": len(rejected_records),
        },
        "synthetic_provenance_not_gold": {
            **synthetic_counts,
            "image_level_provenance_rows": len(provenance),
        },
        "model_predictions": prediction_inventory,
        "normalization": [
            "CSV decoded as UTF-8 with optional BOM",
            "CRLF/CR converted to LF",
            "Unicode normalized to NFC",
            "outer whitespace stripped from annotations",
            "JSON list and QA object shapes validated",
            "annotation wording otherwise unchanged",
        ],
        "image_policy": [
            "all kept references downloaded or reused from cache",
            "source SHA-256 matched downloaded bytes",
            "Pillow container verification",
            "JPEG/PNG only",
            f"maximum encoded bytes {MAX_IMAGE_BYTES}",
            f"maximum pixels {MAX_PIXELS}",
            "content-addressed local storage under images/",
        ],
        "dedup_policy": (
            "Exact equality after normalization across problem_id, image identity, "
            "teacher caption, teacher question, and teacher answer."
        ),
        "cross_dataset_audit": {
            "targets": ["ScratchMath", "ErrorRadar"],
            "source_coverage": cross_source_counts,
            "collision_counts": dict(sorted(collision_counts.items())),
            "total_collision_rows": len(collisions),
            "question_policy": (
                "Exact after NFC, line-ending normalization, outer trim, and "
                "whitespace-run collapse; no fuzzy matching."
            ),
            "image_policy": (
                "SHA-256 byte equality against ScratchMath local images and "
                "downloaded ErrorRadar image URLs."
            ),
            "errorradar_image_urls": (
                {
                    "audited": error_image_audit["unique_urls"],
                    "verified": sum(
                        row["status"] == "verified" for row in error_image_audit["results"]
                    ),
                    "failed": sum(
                        row["status"] != "verified" for row in error_image_audit["results"]
                    ),
                }
                if error_image_audit is not None
                else {"skipped": True}
            ),
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "qc": {
            "row_accounting_balances": (
                len(raw_rows) == len(verified_images) + len(rejected_images)
            ),
            "all_kept_images_verified": all(
                image_check_by_row[row["source_row_index"]]["status"] == "verified"
                for row in verified_images
            ),
            "all_kept_local_image_hashes_match": all(
                sha256_file(output_dir / image_check_by_row[row["source_row_index"]]["path"])
                == row["image_sha256"]
                for row in verified_images
            ),
            "all_gold_records_are_teacher": all(row["gold_source"] == "teacher" for row in records),
            "gold_records_exactly_unique": len(records) == len(seen_records),
            "synthetic_annotations_separate_from_gold": all(
                not any(
                    key in row for key in ("facets_gpt4o", "facets_claude", "qa_gpt4o", "qa_claude")
                )
                for row in records
            ),
            "model_predictions_not_used": True,
        },
    }
    write_json(output_dir / "report.json", report)

    qc = [
        "# DrawEduMath fast-clean QC",
        "",
        f"- Source image rows: **{len(raw_rows)}**",
        f"- Verified/kept image rows: **{len(verified_images)}**",
        f"- Unique verified image SHA-256 values: **{len(unique_image_shas)}**",
        f"- Teacher gold QA source pairs: **{teacher_qa_total}**",
        f"- Teacher gold QA records kept: **{len(records)}**",
        f"- Exact gold duplicates removed: **{len(rejected_records)}**",
        f"- Synthetic provenance rows: **{len(provenance)}**",
        f"- Cross-dataset exact collision rows: **{len(collisions)}**",
        "",
        "## Checks",
        "",
    ]
    qc.extend(f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["qc"].items())
    qc.extend(
        [
            "",
            "Only teacher-authored QA is present in `records.jsonl`. GPT-4o and "
            "Claude annotations are isolated in `provenance.jsonl`; "
            "`DrawEduMath_Model_Predictions.csv` is excluded and not parsed.",
            "",
            "No caption, question, answer, facet, or generated QA text was rewritten.",
            "",
        ]
    )
    (output_dir / "QC.md").write_text("\n".join(qc), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# DrawEduMath fast-clean artifacts\n\n"
        "- `records.jsonl`: deduplicated teacher-gold QA training records.\n"
        "- `provenance.jsonl`: GPT-4o/Claude facets and QA, audit-only.\n"
        "- `image_manifest.jsonl`: per-source-row image URL/SHA/container checks.\n"
        "- `images/`: verified content-addressed local image copies.\n"
        "- `rejected.jsonl`: invalid image rows and exact gold duplicates.\n"
        "- `cross_dataset_collisions.jsonl`: exact ScratchMath/ErrorRadar collisions.\n"
        "- `errorradar_image_audit.jsonl`: downloaded ErrorRadar image hashes/status.\n"
        "- `report.json`, `QC.md`: counts, policies, and invariant checks.\n"
        "- `clean.py`: deterministic reproducible pipeline.\n",
        encoding="utf-8",
    )
    if not all(report["qc"].values()):
        raise RuntimeError("QC failure; inspect report.json")


if __name__ == "__main__":
    main()
