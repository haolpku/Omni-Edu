"""Diagnostic-reasoning cleaning (ErrorRadar, StepVerify, Beetle, SciEntsBank, MathDial)."""

from __future__ import annotations

import sys


import argparse
import ast
import csv
import hashlib
import json
import re
import sys as _sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = pools.source_dir("diagnostic_reasoning")
SOURCE_ROOT = DEFAULT_ROOT / "huggingface"

DATASETS = (
    "ErrorRadar",
    "MAP-Charting-Student-Math-Misunderstandings",
    "StepVerify",
    "algebra_misconceptions",
)


def normalize_text(value: Any, *, optional: bool = False) -> str | None:
    if value is None:
        return None if optional else ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if optional and not text:
        return None
    return text


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


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


def exact_deduplicate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = canonical_json(row)
        if key in seen:
            rejected.append({"reason": "exact_normalized_duplicate", "record": row})
            continue
        seen.add(key)
        kept.append(row)
    return kept, rejected


def base_report(
    dataset: str,
    source_files: list[str],
    input_rows: int,
    kept_rows: int,
    rejected: list[dict[str, Any]],
    split_counts: Counter[str],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scope": "deterministic_quick_clean_no_answer_rewrite_no_new_cot",
        "source_files": source_files,
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_policy": (
            "Preserve source split labels. Data without an official source split "
            "remains `unsplit`; no random split was created."
        ),
        "dedup_policy": "Exact equality after documented field/null normalization.",
    }


def clean_error_radar(out_dir: Path) -> dict[str, Any]:
    source = SOURCE_ROOT / "ErrorRadar" / "ErrorRadar_dataset.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = [
            "question_id",
            "content_image",
            "study_level",
            "question_type",
            "content",
            "answer",
            "user_answer",
            "user_answer_steps",
            "error_step",
            "error_category",
        ]
        if reader.fieldnames != expected:
            raise ValueError(f"ErrorRadar schema mismatch: {reader.fieldnames!r} != {expected!r}")
        source_rows = list(reader)

    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    required = (
        "question_id",
        "content_image",
        "study_level",
        "content",
        "user_answer",
        "user_answer_steps",
        "error_step",
        "error_category",
    )
    for raw in source_rows:
        missing = [field for field in required if not normalize_text(raw[field])]
        if missing:
            rejected.append(
                {
                    "reason": "missing_required_field",
                    "fields": missing,
                    "source_record_id": normalize_text(raw["question_id"]),
                }
            )
            continue
        try:
            answers = ast.literal_eval(raw["answer"])
        except (ValueError, SyntaxError):
            answers = None
        if (
            not isinstance(answers, list)
            or not answers
            or not all(isinstance(item, str) and item.strip() for item in answers)
        ):
            rejected.append(
                {
                    "reason": "invalid_answer_list",
                    "source_record_id": normalize_text(raw["question_id"]),
                }
            )
            continue

        error_match = re.fullmatch(r"Step\s+(\d+)", normalize_text(raw["error_step"]))
        labels = {
            int(value)
            for value in re.findall(
                r"(?im)^\s*Step\s+(\d+)\s*:", normalize_text(raw["user_answer_steps"])
            )
        }
        if error_match is None or int(error_match.group(1)) not in labels:
            rejected.append(
                {
                    "reason": "error_step_not_present_in_numbered_steps",
                    "source_record_id": normalize_text(raw["question_id"]),
                    "error_step": normalize_text(raw["error_step"]),
                    "observed_numbered_steps": sorted(labels),
                }
            )
            continue

        normalized.append(
            {
                "dataset": "ErrorRadar",
                "split": "unsplit",
                "source_record_id": normalize_text(raw["question_id"]),
                "content_image": normalize_text(raw["content_image"]),
                "study_level": normalize_text(raw["study_level"]),
                "question_type": normalize_text(raw["question_type"], optional=True),
                "problem": normalize_text(raw["content"]),
                "reference_answers": [normalize_text(item) for item in answers],
                "student_answer": normalize_text(raw["user_answer"]),
                "student_solution_steps": normalize_text(raw["user_answer_steps"]),
                "first_error_step": normalize_text(raw["error_step"]),
                "error_category": normalize_text(raw["error_category"]),
            }
        )

    kept, duplicate_rejections = exact_deduplicate(normalized)
    rejected.extend(duplicate_rejections)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "ErrorRadar",
        [str(source.relative_to(DEFAULT_ROOT))],
        len(source_rows),
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["schema_checks"] = {
        "source_columns_exact": True,
        "source_question_ids_unique": len({row["question_id"] for row in source_rows})
        == len(source_rows),
        "kept_error_step_present_in_numbered_steps": True,
    }
    report["normalization"] = [
        "CSV UTF-8 BOM handling",
        "snake_case semantic field names",
        "CRLF/CR to LF and outer whitespace trim",
        "blank optional question_type to null",
        "Python-literal answer list parsed to JSON string list",
    ]
    write_json(out_dir / "report.json", report)
    return report


def clean_map_restricted(out_dir: Path) -> dict[str, Any]:
    dataset_dir = SOURCE_ROOT / "MAP-Charting-Student-Math-Misunderstandings"
    restriction = dataset_dir / "DO_NOT_USE_UNTIL_KAGGLE_ACCESS_CONFIRMED.md"
    manifest = DEFAULT_ROOT / "LOCAL_DATASET_MANIFEST.md"
    if not restriction.exists():
        raise FileNotFoundError(f"Missing restriction marker: {restriction}")
    manifest_text = manifest.read_text(encoding="utf-8")
    if (
        "MAP-Charting-Student-Math-Misunderstandings" not in manifest_text
        or "downloaded_quarantined" not in manifest_text
    ):
        raise ValueError("LOCAL_DATASET_MANIFEST.md does not confirm MAP downloaded_quarantined")

    schemas = {
        "train.csv": [
            "row_id",
            "QuestionId",
            "QuestionText",
            "MC_Answer",
            "StudentExplanation",
            "Category",
            "Misconception",
        ],
        "test.csv": [
            "row_id",
            "QuestionId",
            "QuestionText",
            "MC_Answer",
            "StudentExplanation",
        ],
    }
    split_stats: dict[str, Any] = {}
    row_ids_by_split: dict[str, set[str]] = {}
    question_ids_by_split: dict[str, set[str]] = {}
    for filename, expected in schemas.items():
        path = dataset_dir / filename
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected:
                raise ValueError(
                    f"MAP schema mismatch in {filename}: " f"{reader.fieldnames!r} != {expected!r}"
                )
            rows = list(reader)
        split = filename.removesuffix(".csv")
        row_ids = [normalize_text(row["row_id"]) for row in rows]
        question_ids = [normalize_text(row["QuestionId"]) for row in rows]
        row_ids_by_split[split] = set(row_ids)
        question_ids_by_split[split] = set(question_ids)
        split_stats[split] = {
            "rows": len(rows),
            "columns": expected,
            "blank_counts": {
                field: sum(not normalize_text(row[field]) for row in rows) for field in expected
            },
            "unique_row_ids": len(set(row_ids)),
            "row_ids_unique": len(set(row_ids)) == len(row_ids),
            "exact_source_rows": len({tuple(row[field] for field in expected) for row in rows}),
        }

    audited_rows = sum(stats["rows"] for stats in split_stats.values())
    report = {
        "dataset": "MAP-Charting-Student-Math-Misunderstandings",
        "status": "downloaded_quarantined",
        "quarantine_route": "license_access_quarantine",
        "manifest_status": "downloaded_quarantined",
        "manifest_file": str(manifest.relative_to(DEFAULT_ROOT)),
        "audited_source_rows": audited_rows,
        "quarantined_source_rows": audited_rows,
        "kept_rows": 0,
        "usable_records_written": 0,
        "row_level_quarantine_records_written": 0,
        "restriction_marker": str(restriction.relative_to(DEFAULT_ROOT)),
        "reason": (
            "LOCAL_DATASET_MANIFEST.md marks this downloaded Hugging Face copy as "
            "`downloaded_quarantined`: it conflicts with Kaggle rules prohibiting "
            "redistribution to users who have not joined/accepted the competition "
            "rules. The entire dataset is routed to license/access quarantine. "
            "Quick-clean performed aggregate structure/count checks only and emitted "
            "no trainable, evaluable, redistributed, or row-level quarantine data."
        ),
        "source_files_checked": [
            str((dataset_dir / name).relative_to(DEFAULT_ROOT)) for name in schemas
        ],
        "source_split_stats": split_stats,
        "split_checks": {
            "row_id_overlap_train_test": len(row_ids_by_split["train"] & row_ids_by_split["test"]),
            "question_id_overlap_train_test": len(
                question_ids_by_split["train"] & question_ids_by_split["test"]
            ),
            "evaluation_isolation": (
                "Entire dataset quarantined. No source rows copied; source train/test "
                "labels were only counted for structural audit."
            ),
        },
    }
    write_jsonl(out_dir / "kept.jsonl", [])
    write_jsonl(out_dir / "rejected.jsonl", [])
    write_json(out_dir / "report.json", report)
    return report


def valid_dialog(dialog: Any) -> bool:
    return (
        isinstance(dialog, list)
        and bool(dialog)
        and all(
            isinstance(turn, dict)
            and set(turn) == {"text", "user"}
            and bool(normalize_text(turn["text"]))
            and bool(normalize_text(turn["user"]))
            for turn in dialog
        )
    )


def clean_stepverify(out_dir: Path) -> dict[str, Any]:
    source = SOURCE_ROOT / "StepVerify" / "stepverify.json"
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("StepVerify top level must be a list")
    expected = {
        "problem",
        "topic",
        "reference_solution",
        "student_incorrect_solution",
        "incorrect_index",
        "incorrect_step",
        "error_category",
        "error_description",
        "dialog_history",
        "student_correct_response",
    }

    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_index, raw in enumerate(source_rows):
        if not isinstance(raw, dict) or set(raw) != expected:
            rejected.append({"reason": "schema_mismatch", "source_index": source_index})
            continue
        steps = raw["student_incorrect_solution"]
        index = raw["incorrect_index"]
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(step, str) and step.strip() for step in steps)
            or not isinstance(index, int)
            or not 0 <= index < len(steps)
        ):
            rejected.append(
                {"reason": "invalid_solution_steps_or_index", "source_index": source_index}
            )
            continue
        if normalize_text(raw["incorrect_step"]) != normalize_text(steps[index]):
            rejected.append(
                {"reason": "incorrect_step_index_mismatch", "source_index": source_index}
            )
            continue
        if not valid_dialog(raw["dialog_history"]):
            rejected.append({"reason": "invalid_dialog_history", "source_index": source_index})
            continue
        required_text = (
            "problem",
            "topic",
            "reference_solution",
            "incorrect_step",
            "error_category",
            "student_correct_response",
        )
        missing = [field for field in required_text if not normalize_text(raw[field])]
        if missing:
            rejected.append(
                {
                    "reason": "missing_required_field",
                    "fields": missing,
                    "source_index": source_index,
                }
            )
            continue

        content_for_id = {key: raw[key] for key in sorted(expected) if key != "error_description"}
        normalized.append(
            {
                "dataset": "StepVerify",
                "split": "unsplit",
                "source_record_id": stable_id("stepverify", content_for_id),
                "problem": normalize_text(raw["problem"]),
                "topic": normalize_text(raw["topic"]),
                "reference_solution": normalize_text(raw["reference_solution"]),
                "student_incorrect_solution": [
                    normalize_text(step) for step in raw["student_incorrect_solution"]
                ],
                "incorrect_index": index,
                "incorrect_step": normalize_text(raw["incorrect_step"]),
                "error_category": normalize_text(raw["error_category"]),
                "error_description": normalize_text(raw["error_description"], optional=True),
                "dialog_history": [
                    {
                        "user": normalize_text(turn["user"]),
                        "text": normalize_text(turn["text"]),
                    }
                    for turn in raw["dialog_history"]
                ],
                "student_correct_response": normalize_text(raw["student_correct_response"]),
            }
        )

    kept, duplicate_rejections = exact_deduplicate(normalized)
    rejected.extend(duplicate_rejections)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "StepVerify",
        [str(source.relative_to(DEFAULT_ROOT))],
        len(source_rows),
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["schema_checks"] = {
        "top_level_list": True,
        "kept_schema_exact": True,
        "kept_incorrect_index_in_range": True,
        "kept_incorrect_step_matches_index": True,
        "kept_dialog_turns_valid": True,
    }
    report["normalization"] = [
        "CRLF/CR to LF and outer whitespace trim",
        "null/blank optional error_description to null",
        "stable content-hash source_record_id because source has no row id",
    ]
    write_json(out_dir / "report.json", report)
    return report


def clean_algebra_misconceptions(out_dir: Path) -> dict[str, Any]:
    source = SOURCE_ROOT / "algebra_misconceptions" / "data" / "data.json"
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("algebra_misconceptions top level must be a list")
    expected = {
        "Misconception",
        "Misconception ID",
        "Topic",
        "Example Number",
        "Question",
        "Incorrect Answer",
        "Correct Answer",
        "Question image",
        "Learner Answer image",
        "Correct Answer image",
        "Source",
        "Explanation",
    }

    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_index, raw in enumerate(source_rows):
        if not isinstance(raw, dict) or set(raw) != expected:
            rejected.append({"reason": "schema_mismatch", "source_index": source_index})
            continue
        required = (
            "Misconception",
            "Misconception ID",
            "Topic",
            "Question",
            "Incorrect Answer",
            "Correct Answer",
        )
        missing = [field for field in required if not normalize_text(raw[field])]
        if missing:
            rejected.append(
                {
                    "reason": "missing_required_field",
                    "fields": missing,
                    "source_index": source_index,
                }
            )
            continue
        example_number = raw["Example Number"]
        if not isinstance(example_number, int) or example_number not in {1, 2, 3, 4}:
            rejected.append({"reason": "invalid_example_number", "source_index": source_index})
            continue
        misconception_id = normalize_text(raw["Misconception ID"])
        if re.fullmatch(r"MaE\d{2}", misconception_id) is None:
            rejected.append({"reason": "invalid_misconception_id", "source_index": source_index})
            continue

        normalized.append(
            {
                "dataset": "algebra_misconceptions",
                "split": "unsplit",
                "source_record_id": f"{misconception_id}_example_{example_number}",
                "misconception": normalize_text(raw["Misconception"]),
                "misconception_id": misconception_id,
                "topic": normalize_text(raw["Topic"]),
                "example_number": example_number,
                "problem": normalize_text(raw["Question"]),
                "student_incorrect_answer": normalize_text(raw["Incorrect Answer"]),
                "reference_answer": normalize_text(raw["Correct Answer"]),
                "question_image_ref": normalize_text(raw["Question image"], optional=True),
                "learner_answer_image_ref": normalize_text(
                    raw["Learner Answer image"], optional=True
                ),
                "correct_answer_image_ref": normalize_text(
                    raw["Correct Answer image"], optional=True
                ),
                "source_citation": normalize_text(raw["Source"], optional=True),
                "source_explanation": normalize_text(raw["Explanation"], optional=True),
            }
        )

    kept, duplicate_rejections = exact_deduplicate(normalized)
    rejected.extend(duplicate_rejections)
    ids = [row["source_record_id"] for row in kept]
    if len(ids) != len(set(ids)):
        raise ValueError("algebra_misconceptions composite source ids are not unique")
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "algebra_misconceptions",
        [str(source.relative_to(DEFAULT_ROOT))],
        len(source_rows),
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["schema_checks"] = {
        "top_level_list": True,
        "source_composite_ids_unique": True,
        "misconception_count": len({row["misconception_id"] for row in kept}),
        "examples_per_misconception": dict(
            sorted(Counter(row["misconception_id"] for row in kept).items())
        ),
    }
    report["normalization"] = [
        "source field names converted to snake_case semantic names",
        "CRLF/CR to LF and outer whitespace trim",
        "blank optional image/source/explanation fields to null",
        "source_record_id composed from misconception id and example number",
    ]
    write_json(out_dir / "report.json", report)
    return report


def main_batch1() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=DATASETS,
        action="append",
        help="Run only selected dataset(s); default is all four.",
    )
    args = parser.parse_args()
    selected = set(args.only or DATASETS)

    cleaners = {
        "ErrorRadar": clean_error_radar,
        "MAP-Charting-Student-Math-Misunderstandings": clean_map_restricted,
        "StepVerify": clean_stepverify,
        "algebra_misconceptions": clean_algebra_misconceptions,
    }
    summary: dict[str, Any] = {}
    for dataset in DATASETS:
        if dataset not in selected:
            continue
        output_dir = PIPELINE_DIR / dataset / "cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary[dataset] = cleaners[dataset](output_dir)
    print(
        json.dumps(
            {
                name: {
                    "kept_rows": report.get("kept_rows", report.get("usable_records_written", 0)),
                    "rejected_rows": report.get("rejected_rows", 0),
                    "status": report.get("status", "cleaned"),
                }
                for name, report in summary.items()
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


# ------------------------------------------------------------------ batch 2

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = pools.source_dir("diagnostic_reasoning")
SOURCE_ROOT = DEFAULT_ROOT / "huggingface"

LABEL_NAMES = {
    0: "correct",
    1: "contradictory",
    2: "partially_correct_incomplete",
    3: "irrelevant",
    4: "non_domain",
}
VALID_LABELS = set(LABEL_NAMES.values())


def normalize_text(value: Any, *, optional: bool = False) -> str | None:
    if value is None:
        return None if optional else ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if optional and not text:
        return None
    return text


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


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


def load_parquet(path: Path, expected_columns: list[str]) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if table.column_names != expected_columns:
        raise ValueError(
            f"{path} schema mismatch: {table.column_names!r} != " f"{expected_columns!r}"
        )
    return table.to_pylist()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def reject(
    rejected: list[dict[str, Any]],
    reason: str,
    *,
    split: str,
    source_record_id: Any,
    **details: Any,
) -> None:
    entry = {
        "reason": reason,
        "split": split,
        "source_record_id": source_record_id,
    }
    entry.update(details)
    rejected.append(entry)


def exact_semantic_deduplicate(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop exact normalized duplicates within a split, keeping first source row."""
    kept: list[dict[str, Any]] = []
    seen_by_split: dict[str, dict[str, str]] = {}
    for row in rows:
        split = row["split"]
        key = canonical_json([row[field] for field in key_fields])
        seen = seen_by_split.setdefault(split, {})
        if key in seen:
            reject(
                rejected,
                "exact_normalized_semantic_duplicate_within_split",
                split=split,
                source_record_id=row["source_record_id"],
                duplicate_of_source_record_id=seen[key],
            )
            continue
        seen[key] = row["source_record_id"]
        kept.append(row)
    return kept


def isolate_training_from_evaluation(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove exact train records duplicated in any official evaluation split."""
    evaluation_keys = {
        canonical_json([row[field] for field in key_fields])
        for row in rows
        if row["split"] != "train"
    }
    kept = []
    for row in rows:
        key = canonical_json([row[field] for field in key_fields])
        if row["split"] == "train" and key in evaluation_keys:
            reject(
                rejected,
                "exact_train_evaluation_semantic_overlap",
                split="train",
                source_record_id=row["source_record_id"],
            )
            continue
        kept.append(row)
    return kept


def overlap_counts(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[str, int]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        key = canonical_json([row[field] for field in key_fields])
        by_split.setdefault(row["split"], set()).add(key)
    output = {}
    splits = sorted(by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            output[f"{left}__{right}"] = len(by_split[left] & by_split[right])
    return output


def base_report(
    dataset: str,
    source_files: list[str],
    input_rows: int,
    rows_before_dedup: list[dict[str, Any]],
    kept_rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    semantic_key: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "scope": "deterministic_quick_clean_no_answer_rewrite_no_synthesis",
        "source_files": source_files,
        "input_rows": input_rows,
        "kept_rows": len(kept_rows),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in kept_rows).items())),
        "split_policy": (
            "Preserve official source splits and test subsets. Exact duplicate "
            "removal is split-local; when an exact task-semantic record occurs in "
            "train and evaluation, the training copy is removed and evaluation "
            "rows are retained in place."
        ),
        "dedup_policy": (
            "Exact equality after field/null/line-ending normalization on task "
            f"semantic fields: {list(semantic_key)!r}. Source IDs are excluded "
            "from the duplicate key."
        ),
        "cross_split_exact_semantic_overlap_before_dedup": overlap_counts(
            rows_before_dedup, semantic_key
        ),
        "cross_split_exact_semantic_overlap_after_dedup": overlap_counts(kept_rows, semantic_key),
    }


def clean_classification_dataset(
    *,
    dataset: str,
    files: list[tuple[str, str]],
    out_dir: Path,
) -> dict[str, Any]:
    columns = ["id", "question", "reference_answer", "student_answer", "label"]
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    input_rows = 0
    source_files = []
    source_id_counts: dict[str, int] = {}
    label_counts_before: Counter[str] = Counter()

    for split, filename in files:
        path = SOURCE_ROOT / dataset / "data" / filename
        source_files.append(str(path.relative_to(DEFAULT_ROOT)))
        rows = load_parquet(path, columns)
        input_rows += len(rows)
        source_id_counts[split] = len({row["id"] for row in rows})
        for raw in rows:
            source_id = normalize_text(raw["id"])
            missing = [
                field
                for field in ("id", "question", "reference_answer", "student_answer")
                if not normalize_text(raw[field])
            ]
            if missing:
                reject(
                    rejected,
                    "missing_required_field",
                    split=split,
                    source_record_id=source_id,
                    fields=missing,
                )
                continue
            label_id = raw["label"]
            if label_id not in LABEL_NAMES:
                reject(
                    rejected,
                    "invalid_label_id",
                    split=split,
                    source_record_id=source_id,
                    label_id=label_id,
                )
                continue
            label = LABEL_NAMES[label_id]
            label_counts_before[f"{split}:{label}"] += 1
            normalized.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_record_id": source_id,
                    "question": normalize_text(raw["question"]),
                    "reference_answer": normalize_text(raw["reference_answer"]),
                    "student_answer": normalize_text(raw["student_answer"]),
                    "label_5way": label,
                    "label_5way_id": label_id,
                }
            )

    semantic_key = (
        "question",
        "reference_answer",
        "student_answer",
        "label_5way",
    )
    rows_before_dedup = list(normalized)
    kept = exact_semantic_deduplicate(normalized, semantic_key, rejected)
    kept = isolate_training_from_evaluation(kept, semantic_key, rejected)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        dataset,
        source_files,
        input_rows,
        rows_before_dedup,
        kept,
        rejected,
        semantic_key,
    )
    report["schema_checks"] = {
        "source_columns_exact": True,
        "source_ids_unique_within_each_split": all(
            source_id_counts[split] == sum(1 for row in rows_before_dedup if row["split"] == split)
            for split in source_id_counts
        ),
        "kept_labels_valid": all(row["label_5way"] in VALID_LABELS for row in kept),
        "kept_required_fields_nonempty": True,
    }
    report["label_counts_before_filtering"] = dict(sorted(label_counts_before.items()))
    report["label_counts_after_filtering"] = dict(
        sorted(Counter(f"{r['split']}:{r['label_5way']}" for r in kept).items())
    )
    report["normalization"] = [
        "Parquet schema validated exactly",
        "CRLF/CR converted to LF and outer whitespace trimmed",
        "integer ClassLabel converted to explicit label_5way string while retaining id",
        "source id renamed source_record_id",
    ]
    write_json(out_dir / "report.json", report)
    return report


def clean_beetle_atomi(out_dir: Path) -> dict[str, Any]:
    dataset = "Beetle-Atomi-5way"
    columns = [
        "question_id",
        "question",
        "question_qtype",
        "question_module",
        "question_stype",
        "reference_answer",
        "reference_answer_quality",
        "student_answer",
        "label_5way",
        "test_set",
    ]
    files = [
        ("train", "train-00000-of-00001.parquet"),
        ("test", "test-00000-of-00001.parquet"),
    ]
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_files = []
    input_rows = 0
    expected_partitions = {
        "train": {"training"},
        "test": {"unseen-answers", "unseen-questions"},
    }
    observed_partitions: dict[str, Counter[str]] = {}

    for physical_split, filename in files:
        path = SOURCE_ROOT / dataset / "data" / filename
        source_files.append(str(path.relative_to(DEFAULT_ROOT)))
        rows = load_parquet(path, columns)
        input_rows += len(rows)
        observed_partitions[physical_split] = Counter(
            normalize_text(row["test_set"]) for row in rows
        )
        for row_number, raw in enumerate(rows, 1):
            source_id = stable_id(
                "atomi",
                {
                    "physical_split": physical_split,
                    "row_number": row_number,
                    "row": raw,
                },
            )
            missing = [field for field in columns if not normalize_text(raw[field])]
            if missing:
                reject(
                    rejected,
                    "missing_required_field",
                    split=physical_split,
                    source_record_id=source_id,
                    fields=missing,
                )
                continue
            partition = normalize_text(raw["test_set"])
            if partition not in expected_partitions[physical_split]:
                reject(
                    rejected,
                    "test_set_inconsistent_with_physical_split",
                    split=physical_split,
                    source_record_id=source_id,
                    test_set=partition,
                )
                continue
            label = normalize_text(raw["label_5way"])
            if label not in VALID_LABELS:
                reject(
                    rejected,
                    "invalid_label",
                    split=physical_split,
                    source_record_id=source_id,
                    label_5way=label,
                )
                continue
            split = {
                "training": "train",
                "unseen-answers": "test_ua",
                "unseen-questions": "test_uq",
            }[partition]
            normalized.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_record_id": source_id,
                    "question_id": normalize_text(raw["question_id"]),
                    "question": normalize_text(raw["question"]),
                    "question_type": normalize_text(raw["question_qtype"]),
                    "question_module": normalize_text(raw["question_module"]),
                    "question_subtype": normalize_text(raw["question_stype"]),
                    "reference_answer": normalize_text(raw["reference_answer"]),
                    "reference_answer_quality": normalize_text(raw["reference_answer_quality"]),
                    "student_answer": normalize_text(raw["student_answer"]),
                    "label_5way": label,
                    "source_test_set": partition,
                }
            )

    semantic_key = (
        "question_id",
        "question",
        "question_type",
        "question_module",
        "question_subtype",
        "reference_answer",
        "reference_answer_quality",
        "student_answer",
        "label_5way",
    )
    rows_before_dedup = list(normalized)
    kept = exact_semantic_deduplicate(normalized, semantic_key, rejected)
    kept = isolate_training_from_evaluation(kept, semantic_key, rejected)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        dataset,
        source_files,
        input_rows,
        rows_before_dedup,
        kept,
        rejected,
        semantic_key,
    )
    report["schema_checks"] = {
        "source_columns_exact": True,
        "physical_split_test_set_consistent": True,
        "kept_labels_valid": all(row["label_5way"] in VALID_LABELS for row in kept),
        "kept_required_fields_nonempty": True,
    }
    report["source_partition_counts"] = {
        split: dict(sorted(counts.items())) for split, counts in sorted(observed_partitions.items())
    }
    report["label_counts_after_filtering"] = dict(
        sorted(Counter(f"{r['split']}:{r['label_5way']}" for r in kept).items())
    )
    report["normalization"] = [
        "Parquet schema validated exactly",
        "CRLF/CR converted to LF and outer whitespace trimmed",
        "question_qtype/question_stype renamed question_type/question_subtype",
        "official test_set values promoted to isolated test_ua/test_uq splits",
        "stable content-and-source-position id added because source has no row id",
    ]
    write_json(out_dir / "report.json", report)
    return report


def parse_dialogue(conversation: str) -> list[dict[str, Any]] | None:
    turns = []
    for index, raw_turn in enumerate(conversation.split("|EOM|")):
        text = raw_turn.strip()
        if not text or ":" not in text:
            return None
        speaker, utterance = text.split(":", 1)
        speaker = speaker.strip()
        utterance = utterance.strip()
        if not speaker:
            return None
        dialog_act = None
        if speaker == "Teacher":
            match = re.match(r"^\(([^()\n]+)\)\s*(.*)$", utterance, flags=re.S)
            if match:
                dialog_act = normalize_text(match.group(1))
                utterance = normalize_text(match.group(2))
        turns.append(
            {
                "turn_index": index,
                "speaker": speaker,
                "dialog_act": dialog_act,
                "text": utterance,
            }
        )
    return turns


def clean_mathdial(out_dir: Path) -> dict[str, Any]:
    dataset = "MathDial"
    expected_columns = [
        "qid",
        "scenario",
        "question",
        "ground_truth",
        "student_incorrect_solution",
        "student_profile",
        "teacher_described_confusion",
        "self-correctness",
        "self-typical-confusion",
        "self-typical-interactions",
        "conversation",
    ]
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_files = []
    input_rows = 0
    source_qids: dict[str, set[int]] = {}

    for split, filename in (("train", "train.jsonl"), ("test", "test.jsonl")):
        path = SOURCE_ROOT / dataset / filename
        source_files.append(str(path.relative_to(DEFAULT_ROOT)))
        rows = load_jsonl(path)
        input_rows += len(rows)
        source_qids[split] = {row.get("qid") for row in rows}
        for line_number, raw in enumerate(rows, 1):
            source_id = f"{split}:{line_number}"
            if list(raw) != expected_columns:
                reject(
                    rejected,
                    "schema_mismatch",
                    split=split,
                    source_record_id=source_id,
                    observed_columns=list(raw),
                )
                continue
            required = [
                "qid",
                "scenario",
                "question",
                "ground_truth",
                "student_incorrect_solution",
                "student_profile",
                "conversation",
            ]
            missing = [
                field for field in required if raw[field] is None or not normalize_text(raw[field])
            ]
            if missing:
                reject(
                    rejected,
                    "missing_required_field",
                    split=split,
                    source_record_id=source_id,
                    fields=missing,
                )
                continue
            if not isinstance(raw["qid"], int) or not isinstance(raw["scenario"], int):
                reject(
                    rejected,
                    "invalid_identifier_or_scenario_type",
                    split=split,
                    source_record_id=source_id,
                )
                continue
            if raw["scenario"] not in {1, 2, 3, 4, 5}:
                reject(
                    rejected,
                    "scenario_out_of_range",
                    split=split,
                    source_record_id=source_id,
                    scenario=raw["scenario"],
                )
                continue

            annotations = (
                raw["teacher_described_confusion"],
                raw["self-correctness"],
                raw["self-typical-confusion"],
                raw["self-typical-interactions"],
            )
            annotation_nulls = [value is None for value in annotations]
            if any(annotation_nulls) and not all(annotation_nulls):
                reject(
                    rejected,
                    "partial_teacher_annotation_block",
                    split=split,
                    source_record_id=source_id,
                )
                continue
            self_correctness = normalize_text(raw["self-correctness"], optional=True)
            if self_correctness not in {
                None,
                "Yes",
                "Yes, but I had to reveal the answer",
                "No",
            }:
                reject(
                    rejected,
                    "invalid_self_correctness",
                    split=split,
                    source_record_id=source_id,
                    value=self_correctness,
                )
                continue
            likert_values = [
                raw["self-typical-confusion"],
                raw["self-typical-interactions"],
            ]
            if any(
                value is not None
                and (
                    not isinstance(value, (int, float))
                    or float(value) not in {1.0, 2.0, 3.0, 4.0, 5.0}
                )
                for value in likert_values
            ):
                reject(
                    rejected,
                    "invalid_likert_value",
                    split=split,
                    source_record_id=source_id,
                    values=likert_values,
                )
                continue

            conversation = normalize_text(raw["conversation"])
            turns = parse_dialogue(conversation)
            if turns is None:
                reject(
                    rejected,
                    "invalid_conversation_structure",
                    split=split,
                    source_record_id=source_id,
                )
                continue
            profile_name = normalize_text(raw["student_profile"]).split()[0]
            allowed_speakers = {"Teacher", "Student", profile_name}
            unexpected = sorted(
                {turn["speaker"] for turn in turns if turn["speaker"] not in allowed_speakers}
            )
            if unexpected:
                reject(
                    rejected,
                    "unexpected_dialogue_speaker_or_embedded_artifact",
                    split=split,
                    source_record_id=source_id,
                    unexpected_speakers=unexpected,
                )
                continue

            normalized.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "source_record_id": source_id,
                    "problem_id": raw["qid"],
                    "scenario": raw["scenario"],
                    "question": normalize_text(raw["question"]),
                    "ground_truth": normalize_text(raw["ground_truth"]),
                    "student_incorrect_solution": normalize_text(raw["student_incorrect_solution"]),
                    "student_profile": normalize_text(raw["student_profile"]),
                    "teacher_described_confusion": normalize_text(
                        raw["teacher_described_confusion"], optional=True
                    ),
                    "student_self_correctness": self_correctness,
                    "typical_confusion_likert": (
                        int(raw["self-typical-confusion"])
                        if raw["self-typical-confusion"] is not None
                        else None
                    ),
                    "typical_interactions_likert": (
                        int(raw["self-typical-interactions"])
                        if raw["self-typical-interactions"] is not None
                        else None
                    ),
                    "conversation": conversation,
                    "dialogue_turns": turns,
                }
            )

    semantic_key = (
        "problem_id",
        "scenario",
        "question",
        "ground_truth",
        "student_incorrect_solution",
        "student_profile",
        "teacher_described_confusion",
        "student_self_correctness",
        "typical_confusion_likert",
        "typical_interactions_likert",
        "conversation",
    )
    rows_before_dedup = list(normalized)
    kept = exact_semantic_deduplicate(normalized, semantic_key, rejected)
    kept = isolate_training_from_evaluation(kept, semantic_key, rejected)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        dataset,
        source_files,
        input_rows,
        rows_before_dedup,
        kept,
        rejected,
        semantic_key,
    )
    report["schema_checks"] = {
        "source_columns_and_order_exact": True,
        "kept_scenarios_in_1_to_5": True,
        "kept_dialogues_parse_to_speaker_colon_turns": True,
        "kept_speakers_limited_to_teacher_student_or_profile_name": True,
        "teacher_annotation_block_all_null_or_all_present": True,
    }
    report["test_isolation_checks"] = {
        "problem_id_overlap_train_test": len(
            source_qids.get("train", set()) & source_qids.get("test", set())
        ),
        "interpretation": (
            "Problem IDs overlap by source design, but complete supervised dialogue "
            "records remain in their official split and exact semantic overlap is "
            "reported separately."
        ),
    }
    report["null_optional_annotation_rows_after_filtering"] = sum(
        row["teacher_described_confusion"] is None for row in kept
    )
    report["empty_utterance_turns_preserved"] = sum(
        not turn["text"] for row in kept for turn in row["dialogue_turns"]
    )
    report["normalization"] = [
        "JSONL object schema and field order validated exactly",
        "CRLF/CR converted to LF and outer whitespace trimmed",
        "hyphenated annotation fields renamed to snake_case semantic names",
        "complete all-null teacher annotation blocks preserved as null",
        "conversation retained verbatim after line-ending/outer trim and also parsed into dialogue_turns",
    ]
    write_json(out_dir / "report.json", report)
    return report


def run(root: Path) -> dict[str, Any]:
    global DEFAULT_ROOT, SOURCE_ROOT
    DEFAULT_ROOT = root.resolve()
    SOURCE_ROOT = DEFAULT_ROOT / "huggingface"
    reports = {
        "Beetle": clean_classification_dataset(
            dataset="Beetle",
            files=[
                ("train", "train-00001.parquet"),
                ("test_ua", "test-ua-00001.parquet"),
                ("test_uq", "test-uq-00001.parquet"),
            ],
        out_dir=PIPELINE_DIR / "Beetle" / "cleaned",
        ),
        "Beetle-Atomi-5way": clean_beetle_atomi(PIPELINE_DIR / "Beetle-Atomi-5way" / "cleaned"),
        "SciEntsBank": clean_classification_dataset(
            dataset="SciEntsBank",
            files=[
                ("train", "train-00001.parquet"),
                ("test_ua", "test-ua-00001.parquet"),
                ("test_uq", "test-uq-00001.parquet"),
                ("test_ud", "test-ud-00001.parquet"),
            ],
        out_dir=PIPELINE_DIR / "SciEntsBank" / "cleaned",
        ),
        "MathDial": clean_mathdial(PIPELINE_DIR / "MathDial" / "cleaned"),
    }
    return reports


def main_batch2() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="diagnostic_reasoning dataset root",
    )
    args = parser.parse_args()
    reports = run(args.root)
    for dataset, report in reports.items():
        print(
            f"{dataset}: input={report['input_rows']} "
            f"kept={report['kept_rows']} rejected={report['rejected_rows']}"
        )

def main() -> None:
    argv = list(sys.argv[1:])
    batch = "all"
    if "--batch" in argv:
        index = argv.index("--batch")
        batch = argv[index + 1]
        del argv[index : index + 2]
    sys.argv = [sys.argv[0]] + argv
    runners = {"1": main_batch1, "2": main_batch2}
    if batch == "all":
        for runner in runners.values():
            runner()
        return
    if batch not in runners:
        raise SystemExit(f"unknown batch: {batch} (use all, 1 or 2)")
    runners[batch]()


if __name__ == "__main__":
    main()
