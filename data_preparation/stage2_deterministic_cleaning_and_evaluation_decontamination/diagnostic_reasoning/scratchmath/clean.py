#!/usr/bin/env python3
"""Deterministic fast-clean for the local ScratchMath parquet files.

The scope is deliberately narrow:
* normalize only the original primary/middle fields;
* safely externalize embedded PNG bytes to content-addressed files;
* reject records missing required fields or failing basic integrity checks;
* remove exact duplicates after normalization;
* emit machine-readable reports and QC.

This script never rewrites answers/explanations and never calls a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

import sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import pools
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = pools.source_dir("diagnostic_reasoning") / "huggingface" / "ScratchMath"
LEVELS = ("primary", "middle")
SOURCE_COLUMNS = (
    "question_id",
    "question",
    "answer",
    "solution",
    "student_answer",
    "student_scratchwork",
    "error_category",
    "error_explanation",
)
TEXT_FIELDS = (
    "question_id",
    "question",
    "answer",
    "solution",
    "student_answer",
    "error_explanation",
)
REQUIRED_FIELDS = SOURCE_COLUMNS
ERROR_CATEGORY_LABELS = {
    0: "计算错误",
    1: "题目理解错误",
    2: "知识点错误",
    3: "答题技巧错误",
    4: "手写誊抄错误",
    5: "逻辑推理错误",
    6: "注意力与细节错误",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_IMAGE_BYTES = 50 * 1024 * 1024
MAX_DIMENSION = 100_000
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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_png(data: bytes) -> tuple[int, int]:
    """Strictly validate PNG container structure/CRC without decoding pixels."""
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image_too_large")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("invalid_png_signature")

    offset = len(PNG_SIGNATURE)
    chunk_index = 0
    width = height = 0
    saw_idat = False
    saw_iend = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise ValueError("truncated_png_chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ValueError("truncated_png_chunk_data")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : chunk_end])[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("png_crc_mismatch")

        if chunk_index == 0:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("invalid_png_ihdr")
            width, height = struct.unpack(">II", payload[:8])
            bit_depth, color_type, compression, filtering, interlace = payload[8:]
            valid_bit_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                color_type not in valid_bit_depths
                or bit_depth not in valid_bit_depths[color_type]
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise ValueError("invalid_png_ihdr_values")
            if (
                width <= 0
                or height <= 0
                or width > MAX_DIMENSION
                or height > MAX_DIMENSION
                or width * height > MAX_PIXELS
            ):
                raise ValueError("unsafe_png_dimensions")
        elif chunk_type == b"IHDR":
            raise ValueError("duplicate_png_ihdr")

        if chunk_type == b"IDAT":
            saw_idat = True
        if chunk_type == b"IEND":
            if length != 0:
                raise ValueError("invalid_png_iend")
            saw_iend = True
            offset = chunk_end
            if offset != len(data):
                raise ValueError("trailing_bytes_after_png_iend")
            break

        offset = chunk_end
        chunk_index += 1

    if not saw_idat:
        raise ValueError("missing_png_idat")
    if not saw_iend:
        raise ValueError("missing_png_iend")
    return width, height


def write_content_addressed_image(image_root: Path, data: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    relative_path = Path("images") / digest[:2] / f"{digest}.png"
    destination = image_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise RuntimeError(f"content-address collision at {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"post-write hash mismatch for {temporary}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    width, height = validate_png(data)
    return {
        "path": relative_path.as_posix(),
        "sha256": digest,
        "media_type": "image/png",
        "bytes": len(data),
        "width": width,
        "height": height,
    }


def image_bytes_from_source(value: Any) -> bytes:
    if not isinstance(value, dict):
        raise ValueError("invalid_image_struct")
    data = value.get("bytes")
    source_path = value.get("path")
    if not isinstance(data, bytes) or not data:
        if source_path:
            raise ValueError("path_only_image_not_embedded")
        raise ValueError("missing_image_bytes")
    return data


def source_fingerprint(level: str, row_index: int, raw: dict[str, Any]) -> str:
    safe = {
        "level": level,
        "row_index": row_index,
        "question_id": normalize_text(raw.get("question_id")),
    }
    return hashlib.sha256(canonical_json(safe).encode("utf-8")).hexdigest()[:20]


def normalize_record(
    level: str,
    raw: dict[str, Any],
    image_root: Path,
) -> dict[str, Any]:
    normalized_text = {field: normalize_text(raw.get(field)) for field in TEXT_FIELDS}
    missing = [field for field, value in normalized_text.items() if not value]
    if raw.get("error_category") is None:
        missing.append("error_category")
    if raw.get("student_scratchwork") is None:
        missing.append("student_scratchwork")
    if missing:
        raise ValueError("missing_required_field:" + ",".join(sorted(missing)))

    category = raw["error_category"]
    if isinstance(category, bool) or not isinstance(category, int):
        raise ValueError("invalid_error_category_type")
    if category not in ERROR_CATEGORY_LABELS:
        raise ValueError("invalid_error_category_value")

    image_data = image_bytes_from_source(raw["student_scratchwork"])
    validate_png(image_data)
    image_reference = write_content_addressed_image(image_root, image_data)

    return {
        "subset": level,
        "split": "train",
        "question_id": normalized_text["question_id"],
        "question": normalized_text["question"],
        "answer": normalized_text["answer"],
        "solution": normalized_text["solution"],
        "student_answer": normalized_text["student_answer"],
        "student_scratchwork": image_reference,
        "error_category": category,
        "error_explanation": normalized_text["error_explanation"],
    }


def dedup_key(record: dict[str, Any]) -> str:
    # Ignore only pipeline-added source location metadata; all normalized original
    # fields, including question_id and image hash/metadata, participate.
    original_normalized = {field: record[field] for field in SOURCE_COLUMNS}
    return canonical_json(original_normalized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=pools.cleaned_dir("diagnostic_reasoning", "ScratchMath"))
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    expected_schema = pa.schema(
        [
            pa.field("question_id", pa.string()),
            pa.field("question", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("solution", pa.string()),
            pa.field("student_answer", pa.string()),
            pa.field(
                "student_scratchwork",
                pa.struct(
                    [
                        pa.field("bytes", pa.binary()),
                        pa.field("path", pa.string()),
                    ]
                ),
            ),
            pa.field("error_category", pa.int64()),
            pa.field("error_explanation", pa.string()),
        ]
    )

    source_files: list[str] = []
    source_counts: Counter[str] = Counter()
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for level in LEVELS:
        source = dataset_root / level / "data-00000-of-00001.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        source_files.append(str(source))
        parquet = pq.ParquetFile(source)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError(
                f"{level} schema mismatch:\n{parquet.schema_arrow}\n!=\n" f"{expected_schema}"
            )
        rows = parquet.read().to_pylist()
        source_counts[level] = len(rows)
        for row_index, raw in enumerate(rows):
            try:
                normalized.append(normalize_record(level, raw, output_dir))
            except (ValueError, RuntimeError) as exc:
                reason, _, detail = str(exc).partition(":")
                item: dict[str, Any] = {
                    "reason": reason,
                    "subset": level,
                    "source_row_index": row_index,
                    "source_fingerprint": source_fingerprint(level, row_index, raw),
                    "question_id": normalize_text(raw.get("question_id")) or None,
                }
                if detail:
                    item["detail"] = detail
                rejected.append(item)

    kept: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for record in normalized:
        key = dedup_key(record)
        if key in seen:
            rejected.append(
                {
                    "reason": "exact_normalized_duplicate",
                    "subset": record["subset"],
                    "question_id": record["question_id"],
                    "duplicate_of": {
                        "subset": seen[key]["subset"],
                        "question_id": seen[key]["question_id"],
                    },
                }
            )
        else:
            seen[key] = record
            kept.append(record)

    image_manifest_by_hash = {
        row["student_scratchwork"]["sha256"]: row["student_scratchwork"] for row in kept
    }
    image_manifest = [image_manifest_by_hash[key] for key in sorted(image_manifest_by_hash)]

    write_jsonl(output_dir / "kept.jsonl", kept)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_jsonl(output_dir / "image_manifest.jsonl", image_manifest)

    kept_counts = Counter(row["subset"] for row in kept)
    category_counts = {
        level: {
            str(category): {
                "label": ERROR_CATEGORY_LABELS[category],
                "count": sum(
                    1
                    for row in kept
                    if row["subset"] == level and row["error_category"] == category
                ),
            }
            for category in ERROR_CATEGORY_LABELS
        }
        for level in LEVELS
    }
    rejection_counts = Counter(item["reason"] for item in rejected)
    total_image_bytes = sum(item["bytes"] for item in image_manifest)
    report = {
        "dataset": "ScratchMath",
        "scope": ("deterministic_clean_original_fields_only_no_rewrite_no_model"),
        "source_files": source_files,
        "source_rows": {
            **dict(source_counts),
            "total": sum(source_counts.values()),
        },
        "kept_rows": {
            **dict(kept_counts),
            "total": len(kept),
        },
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "required_fields": list(REQUIRED_FIELDS),
        "normalization": [
            "text values converted to strings only where non-null",
            "CRLF/CR converted to LF",
            "Unicode normalized to NFC",
            "outer whitespace stripped",
            "error_category retained as source integer 0..6",
            "student_scratchwork bytes replaced by content-addressed PNG reference",
        ],
        "image_policy": {
            "storage": "images/<sha256-prefix>/<sha256>.png",
            "reference_path_base": "this output directory",
            "validation": [
                "PNG signature",
                "complete chunk framing",
                "CRC for every chunk",
                "single leading IHDR with valid values",
                "IDAT and terminal IEND required",
                "no bytes after IEND",
                f"maximum encoded bytes {MAX_IMAGE_BYTES}",
                f"maximum dimension {MAX_DIMENSION}",
                f"maximum pixels {MAX_PIXELS}",
                "SHA-256 verified after atomic write",
            ],
            "unique_images": len(image_manifest),
            "total_encoded_bytes": total_image_bytes,
        },
        "dedup_policy": (
            "Global exact equality across all normalized original fields; "
            "student_scratchwork equality uses the full stable reference metadata."
        ),
        "split_policy": ("Both source configs declare only train; no new partition was created."),
        "category_counts": category_counts,
        "qc": {
            "source_schema_exact": True,
            "all_kept_required_fields_present": all(
                all(row.get(field) not in (None, "") for field in REQUIRED_FIELDS) for row in kept
            ),
            "all_kept_categories_valid": all(
                row["error_category"] in ERROR_CATEGORY_LABELS for row in kept
            ),
            "all_kept_image_references_unique_or_shared_by_hash": (
                len(image_manifest_by_hash) == len(image_manifest)
            ),
            "all_referenced_images_exist": all(
                (output_dir / item["path"]).is_file() for item in image_manifest
            ),
            "all_referenced_image_hashes_match": all(
                hashlib.sha256((output_dir / item["path"]).read_bytes()).hexdigest()
                == item["sha256"]
                for item in image_manifest
            ),
            "kept_records_exactly_unique": (len({dedup_key(row) for row in kept}) == len(kept)),
            "row_accounting_balances": (sum(source_counts.values()) == len(kept) + len(rejected)),
        },
    }
    write_json(output_dir / "report.json", report)

    qc_lines = [
        "# ScratchMath fast-clean QC",
        "",
        f"- Source rows: **{sum(source_counts.values())}** "
        f"(primary {source_counts['primary']}, middle {source_counts['middle']})",
        f"- Kept rows: **{len(kept)}** "
        f"(primary {kept_counts['primary']}, middle {kept_counts['middle']})",
        f"- Rejected rows: **{len(rejected)}**",
        f"- Unique externalized PNGs: **{len(image_manifest)}**",
        f"- Externalized encoded bytes: **{total_image_bytes}**",
        "",
        "## Rejection reasons",
        "",
    ]
    if rejection_counts:
        qc_lines.extend(
            f"- `{reason}`: {count}" for reason, count in sorted(rejection_counts.items())
        )
    else:
        qc_lines.append("- None")
    qc_lines.extend(["", "## Checks", ""])
    qc_lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`" for name, passed in report["qc"].items()
    )
    qc_lines.extend(
        [
            "",
            "No answer, solution, student answer, error label, or error "
            "explanation was rewritten or model-generated.",
            "",
        ]
    )
    (output_dir / "QC.md").write_text("\n".join(qc_lines), encoding="utf-8")

    if not all(report["qc"].values()):
        raise RuntimeError("one or more QC checks failed; inspect report.json")


if __name__ == "__main__":
    main()
