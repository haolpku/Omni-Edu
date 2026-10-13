"""Pedagogical-action cleaning (CIMA, TalkMoves, MultiHint, SocraticMATH, EduAdapt)."""

from __future__ import annotations

import sys


import argparse
import csv
import hashlib
import json
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
DEFAULT_ROOT = pools.source_dir("pedagogical_action")
DATASETS = ("CIMA", "TalkMoves", "MultiHint")
EXCLUDED_DATASETS = ("Eedi-QATD-2k",)

CIMA_TUTOR_ACTIONS = (
    "question",
    "hint_or_information_reveal",
    "correction",
    "confirmation",
    "other",
)
CIMA_STUDENT_ACTIONS = ("guess", "question", "affirmation", "other")
TALKMOVES_LABELS = {
    "teacher": set(range(7)),
    "student": set(range(5)),
}


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
    rows: list[dict[str, Any]], key_fields: tuple[str, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for row in rows:
        key = canonical_json({field: row[field] for field in key_fields})
        if key in seen:
            rejected.append(
                {
                    "reason": "exact_normalized_duplicate",
                    "record_id": row["record_id"],
                    "duplicate_of": seen[key],
                }
            )
            continue
        seen[key] = row["record_id"]
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
        "scope": (
            "structure_validation_field_normalization_empty_or_role_filtering_"
            "exact_dedup_no_rewrite_no_generation"
        ),
        "source_files": source_files,
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(item["reason"] for item in rejected).items())),
        "split_counts": dict(sorted(split_counts.items())),
        "split_policy": (
            "Preserve official source split labels. Sources without an official "
            "split remain `unsplit`; no random split is created."
        ),
        "dedup_policy": (
            "Exact equality after documented normalization, within each dataset "
            "and official split/task partition."
        ),
    }


def parse_bool_strings(values: Any, expected_length: int) -> list[bool] | None:
    if not isinstance(values, list) or len(values) != expected_length:
        return None
    result: list[bool] = []
    for value in values:
        if value is True or value == "True":
            result.append(True)
        elif value is False or value == "False":
            result.append(False)
        else:
            return None
    return result


def clean_cima(out_dir: Path) -> dict[str, Any]:
    source = DEFAULT_ROOT / "github" / "CIMA" / "dataset.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if set(payload) != {"prepDataset", "shapeDataset"}:
        raise ValueError(f"CIMA top-level schema mismatch: {sorted(payload)}")
    if payload["shapeDataset"] != {}:
        raise ValueError("CIMA shapeDataset is unexpectedly non-empty")
    source_rows = payload["prepDataset"]
    if not isinstance(source_rows, dict):
        raise TypeError("CIMA prepDataset must be an object keyed by record id")
    if list(source_rows) != [str(i) for i in range(len(source_rows))]:
        raise ValueError("CIMA record ids are not the expected contiguous strings")

    expected_fields = {
        "past_convo",
        "img",
        "prep",
        "engPrep",
        "obj",
        "engObj",
        "color",
        "engColor",
        "grammarRules",
        "studentActions",
        "tutorResponses",
        "tutorActions",
        "tutorKeys",
    }
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source_id, raw in source_rows.items():
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            rejected.append(
                {
                    "reason": "source_schema_mismatch",
                    "source_record_id": source_id,
                    "observed_fields": sorted(raw) if isinstance(raw, dict) else None,
                }
            )
            continue

        dialogue = raw["past_convo"]
        responses = raw["tutorResponses"]
        tutor_actions_raw = raw["tutorActions"]
        student_actions = parse_bool_strings(raw["studentActions"], 4)
        if not isinstance(dialogue, list) or not dialogue:
            rejected.append({"reason": "missing_dialogue", "source_record_id": source_id})
            continue
        if not all(isinstance(turn, str) and normalize_text(turn) for turn in dialogue):
            rejected.append(
                {
                    "reason": "empty_or_invalid_dialogue_turn",
                    "source_record_id": source_id,
                }
            )
            continue
        if len(dialogue) % 2:
            rejected.append(
                {
                    "reason": "role_sequence_not_complete_tutor_student_pairs",
                    "source_record_id": source_id,
                    "turn_count": len(dialogue),
                }
            )
            continue
        if (
            not isinstance(responses, list)
            or not responses
            or not all(
                isinstance(response, str) and normalize_text(response) for response in responses
            )
        ):
            rejected.append(
                {
                    "reason": "empty_or_invalid_tutor_response",
                    "source_record_id": source_id,
                }
            )
            continue
        if not isinstance(tutor_actions_raw, list) or len(tutor_actions_raw) != len(responses):
            rejected.append(
                {
                    "reason": "tutor_response_action_count_mismatch",
                    "source_record_id": source_id,
                }
            )
            continue
        tutor_actions = [parse_bool_strings(action, 5) for action in tutor_actions_raw]
        if any(action is None for action in tutor_actions):
            rejected.append(
                {
                    "reason": "invalid_tutor_action_vector",
                    "source_record_id": source_id,
                }
            )
            continue
        if student_actions is None:
            rejected.append(
                {
                    "reason": "invalid_student_action_vector",
                    "source_record_id": source_id,
                }
            )
            continue

        try:
            image_path = json.loads(raw["img"])
        except (json.JSONDecodeError, TypeError):
            rejected.append({"reason": "invalid_image_path", "source_record_id": source_id})
            continue
        if not isinstance(image_path, str) or not normalize_text(image_path):
            rejected.append({"reason": "invalid_image_path", "source_record_id": source_id})
            continue
        if not (DEFAULT_ROOT / "github" / "CIMA" / image_path).is_file():
            rejected.append(
                {
                    "reason": "missing_referenced_image",
                    "source_record_id": source_id,
                    "image_path": image_path,
                }
            )
            continue

        messages = [
            {
                "role": "tutor" if index % 2 == 0 else "student",
                "content": normalize_text(turn),
            }
            for index, turn in enumerate(dialogue)
        ]
        tutor_candidates = [
            {
                "content": normalize_text(response),
                "actions": {
                    name: value for name, value in zip(CIMA_TUTOR_ACTIONS, action, strict=True)
                },
            }
            for response, action in zip(responses, tutor_actions, strict=True)
        ]
        row = {
            "dataset": "CIMA",
            "split": "unsplit",
            "source_record_id": source_id,
            "image_path": normalize_text(image_path),
            "messages": messages,
            "student_actions": {
                name: value
                for name, value in zip(CIMA_STUDENT_ACTIONS, student_actions, strict=True)
            },
            "tutor_candidates": tutor_candidates,
            "grounding": {
                "italian_preposition": normalize_text(raw["prep"]),
                "english_preposition": normalize_text(raw["engPrep"]),
                "italian_object": normalize_text(raw["obj"]),
                "english_object": normalize_text(raw["engObj"]),
                "italian_color": normalize_text(raw["color"]),
                "english_color": normalize_text(raw["engColor"]),
                "grammar_rules_raw": normalize_text(raw["grammarRules"], optional=True),
            },
        }
        row["record_id"] = stable_id(
            "cima",
            {
                "split": row["split"],
                "messages": row["messages"],
                "student_actions": row["student_actions"],
                "tutor_candidates": row["tutor_candidates"],
                "grounding": row["grounding"],
                "image_path": row["image_path"],
            },
        )
        normalized.append(row)

    dedup_fields = (
        "split",
        "image_path",
        "messages",
        "student_actions",
        "tutor_candidates",
        "grounding",
    )
    kept, duplicate_rejections = exact_deduplicate(normalized, dedup_fields)
    rejected.extend(duplicate_rejections)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "CIMA",
        [str(source.relative_to(DEFAULT_ROOT))],
        len(source_rows),
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["schema_checks"] = {
        "top_level_keys_exact": True,
        "prep_dataset_ids_contiguous": True,
        "shape_dataset_empty": True,
        "accepted_dialogues_alternate_tutor_student": True,
        "accepted_tutor_response_action_counts_match": True,
        "accepted_referenced_images_exist": True,
    }
    report["normalization"] = [
        "CRLF/CR to LF and outer whitespace trim",
        "past_convo converted to messages with source-defined tutor/student alternation",
        "student/tutor action vectors converted to named boolean maps",
        "JSON-quoted image path decoded",
        "empty tutorKeys omitted",
        "grammarRules retained verbatim as grammar_rules_raw because some source rows contain concatenated JSON values",
    ]
    write_json(out_dir / "report.json", report)
    return report


def parse_talkmoves_label(value: Any) -> int | None:
    text = normalize_text(value)
    try:
        number = float(text)
    except ValueError:
        return None
    integer = int(number)
    if number != integer:
        return None
    return integer


def clean_talkmoves(out_dir: Path) -> dict[str, Any]:
    source_dir = DEFAULT_ROOT / "github" / "TalkMoves" / "data"
    source_specs = (
        ("train_teacher.tsv", "train", "teacher"),
        ("test_teacher.tsv", "test", "teacher"),
        ("train_student.tsv", "train", "student"),
        ("test_student.tsv", "test", "student"),
    )
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    input_rows = 0
    source_files: list[str] = []
    source_row_counts: dict[str, int] = {}
    for filename, split, target_role in source_specs:
        source = source_dir / filename
        source_files.append(str(source.relative_to(DEFAULT_ROOT)))
        file_rows = 0
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames != ["text_a", "text_b", "labels"]:
                raise ValueError(f"TalkMoves schema mismatch in {filename}: {reader.fieldnames}")
            for line_number, raw in enumerate(reader, start=2):
                input_rows += 1
                file_rows += 1
                context = normalize_text(raw["text_a"], optional=True)
                utterance = normalize_text(raw["text_b"], optional=True)
                label = parse_talkmoves_label(raw["labels"])
                source_locator = f"{filename}:{line_number}"
                if utterance is None:
                    rejected.append(
                        {
                            "reason": "empty_target_utterance",
                            "source_locator": source_locator,
                        }
                    )
                    continue
                if label not in TALKMOVES_LABELS[target_role]:
                    rejected.append(
                        {
                            "reason": "invalid_role_label",
                            "source_locator": source_locator,
                            "target_role": target_role,
                            "label": normalize_text(raw["labels"], optional=True),
                        }
                    )
                    continue
                context_role = None
                if context is not None:
                    context_role = "student" if target_role == "teacher" else "teacher"
                row = {
                    "dataset": "TalkMoves",
                    "split": split,
                    "task": f"{target_role}_talk_move",
                    "source_locator": source_locator,
                    "context": (
                        None if context is None else {"role": context_role, "content": context}
                    ),
                    "utterance": {"role": target_role, "content": utterance},
                    "label_id": label,
                }
                row["record_id"] = stable_id(
                    "talkmoves",
                    {
                        "split": split,
                        "task": row["task"],
                        "context": row["context"],
                        "utterance": row["utterance"],
                        "label_id": label,
                    },
                )
                normalized.append(row)
        source_row_counts[filename] = file_rows

    dedup_fields = (
        "split",
        "task",
        "context",
        "utterance",
        "label_id",
    )
    kept, duplicate_rejections = exact_deduplicate(normalized, dedup_fields)
    rejected.extend(duplicate_rejections)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "TalkMoves",
        source_files,
        input_rows,
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["task_counts"] = dict(sorted(Counter(row["task"] for row in kept).items()))
    report["split_task_counts"] = {
        f"{split}/{task}": count
        for (split, task), count in sorted(
            Counter((row["split"], row["task"]) for row in kept).items()
        )
    }
    report["source_row_counts"] = source_row_counts
    report["schema_checks"] = {
        "all_tsv_columns_exact": True,
        "teacher_labels_within_0_to_6": True,
        "student_labels_within_0_to_4": True,
        "accepted_target_roles_match_source_task": True,
    }
    report["normalization"] = [
        "TSV UTF-8 BOM handling",
        "CRLF/CR to LF and outer whitespace trim",
        "text_a mapped to optional opposite-role context",
        "text_b mapped to role-explicit target utterance",
        "integral float label strings converted to integers",
        "blank context retained as null because source construction explicitly permits no preceding opposite-role utterance",
    ]
    write_json(out_dir / "report.json", report)
    return report


def clean_multihint(out_dir: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read MultiHint") from exc

    source_dir = DEFAULT_ROOT / "huggingface" / "MultiHint" / "data"
    source_specs = (
        ("train-00000-of-00001.parquet", "train"),
        ("test-00000-of-00001.parquet", "test"),
    )
    expected_schema = {
        "passage": "string",
        "language": "string",
        "question": "string",
        "answer": "string",
        "hint": "string",
        "qa_loop_index": "int64",
        "hint_loop_index": "int64",
        "recovery_count": "int64",
    }
    normalized: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_files: list[str] = []
    source_row_counts: dict[str, int] = {}
    input_rows = 0
    for filename, split in source_specs:
        source = source_dir / filename
        source_files.append(str(source.relative_to(DEFAULT_ROOT)))
        table = pq.read_table(source)
        observed_schema = {field.name: str(field.type) for field in table.schema}
        if observed_schema != expected_schema:
            raise ValueError(f"MultiHint schema mismatch in {filename}: {observed_schema}")
        rows = table.to_pylist()
        source_row_counts[filename] = len(rows)
        for row_number, raw in enumerate(rows):
            input_rows += 1
            source_locator = f"{filename}:{row_number}"
            text_fields = ("passage", "language", "question", "answer", "hint")
            missing = [field for field in text_fields if not normalize_text(raw[field])]
            if missing:
                rejected.append(
                    {
                        "reason": "missing_required_field",
                        "source_locator": source_locator,
                        "fields": missing,
                    }
                )
                continue
            if raw["language"] not in {"English", "Italian", "Farsi"}:
                rejected.append(
                    {
                        "reason": "invalid_language",
                        "source_locator": source_locator,
                        "language": raw["language"],
                    }
                )
                continue
            integer_fields = ("qa_loop_index", "hint_loop_index", "recovery_count")
            if any(not isinstance(raw[field], int) or raw[field] < 0 for field in integer_fields):
                rejected.append(
                    {
                        "reason": "invalid_pipeline_counter",
                        "source_locator": source_locator,
                    }
                )
                continue
            row = {
                "dataset": "MultiHint",
                "split": split,
                "source_locator": source_locator,
                "language": normalize_text(raw["language"]),
                "passage": normalize_text(raw["passage"]),
                "question": normalize_text(raw["question"]),
                "answer": normalize_text(raw["answer"]),
                "hint": normalize_text(raw["hint"]),
                "source_metadata": {
                    "qa_loop_index": raw["qa_loop_index"],
                    "hint_loop_index": raw["hint_loop_index"],
                    "recovery_count": raw["recovery_count"],
                },
            }
            row["record_id"] = stable_id(
                "multihint",
                {
                    "split": split,
                    "language": row["language"],
                    "passage": row["passage"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "hint": row["hint"],
                    "source_metadata": row["source_metadata"],
                },
            )
            normalized.append(row)

    dedup_fields = (
        "split",
        "language",
        "passage",
        "question",
        "answer",
        "hint",
        "source_metadata",
    )
    kept, duplicate_rejections = exact_deduplicate(normalized, dedup_fields)
    rejected.extend(duplicate_rejections)
    write_jsonl(out_dir / "kept.jsonl", kept)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = base_report(
        "MultiHint",
        source_files,
        input_rows,
        len(kept),
        rejected,
        Counter(row["split"] for row in kept),
    )
    report["source_row_counts"] = source_row_counts
    report["language_counts"] = dict(sorted(Counter(row["language"] for row in kept).items()))
    report["split_language_counts"] = {
        f"{split}/{language}": count
        for (split, language), count in sorted(
            Counter((row["split"], row["language"]) for row in kept).items()
        )
    }
    report["schema_checks"] = {
        "parquet_schema_exact": True,
        "accepted_languages_known": True,
        "accepted_pipeline_counters_nonnegative_integers": True,
    }
    report["normalization"] = [
        "CRLF/CR to LF and outer whitespace trim",
        "existing passage/question/answer/hint text retained without rewriting",
        "pipeline counters grouped under source_metadata",
    ]
    write_json(out_dir / "report.json", report)
    return report


def verify_outputs(out_root: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for report in reports:
        dataset = report["dataset"]
        dataset_dir = out_root / dataset / "cleaned"
        records: list[dict[str, Any]] = []
        with (dataset_dir / "kept.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))
        with (dataset_dir / "rejected.jsonl").open(encoding="utf-8") as handle:
            rejected_lines = sum(1 for _ in handle)
        record_lines = len(records)
        record_ids_unique = len({row["record_id"] for row in records}) == record_lines
        dataset_specific = True
        if dataset == "CIMA":
            dataset_specific = all(
                row["split"] == "unsplit"
                and row["messages"]
                and all(
                    turn["role"] == ("tutor" if index % 2 == 0 else "student")
                    and bool(turn["content"])
                    for index, turn in enumerate(row["messages"])
                )
                and all(candidate["content"] for candidate in row["tutor_candidates"])
                for row in records
            )
        elif dataset == "TalkMoves":
            dataset_specific = all(
                row["split"] in {"train", "test"}
                and row["utterance"]["content"]
                and row["utterance"]["role"] in {"teacher", "student"}
                and row["task"] == f'{row["utterance"]["role"]}_talk_move'
                and (row["context"] is None or row["context"]["role"] != row["utterance"]["role"])
                for row in records
            )
        elif dataset == "MultiHint":
            dataset_specific = all(
                row["split"] in {"train", "test"}
                and row["language"] in {"English", "Italian", "Farsi"}
                and all(row[field] for field in ("passage", "question", "answer", "hint"))
                for row in records
            )
        checks.append(
            {
                "dataset": dataset,
                "records_match_report": record_lines == report["kept_rows"],
                "rejections_match_report": rejected_lines == report["rejected_rows"],
                "count_conservation": report["input_rows"]
                == report["kept_rows"] + report["rejected_rows"],
                "record_ids_unique": record_ids_unique,
                "dataset_specific_invariants": dataset_specific,
            }
        )
    passed = all(value for check in checks for key, value in check.items() if key != "dataset")
    return {
        "passed": passed,
        "datasets": list(DATASETS),
        "excluded_datasets": list(EXCLUDED_DATASETS),
        "checks": checks,
        "eedi_qatd_2k_touched": False,
    }


def main_batch1() -> None:
    global DEFAULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="stage 1 source directory")
    parser.add_argument(
        "--out",
        type=Path,
        default=PIPELINE_DIR,
        help="parent of the dataset directories (defaults to stage2/pedagogical_action)",
    )
    args = parser.parse_args()
    DEFAULT_ROOT = args.root.resolve()
    out_root = args.out.resolve()

    reports = [
        clean_cima(out_root / "CIMA" / "cleaned"),
        clean_talkmoves(out_root / "TalkMoves" / "cleaned"),
        clean_multihint(out_root / "MultiHint" / "cleaned"),
    ]
    summary = {
        "datasets": list(DATASETS),
        "excluded_datasets": list(EXCLUDED_DATASETS),
        "totals": {
            "input_rows": sum(report["input_rows"] for report in reports),
            "kept_rows": sum(report["kept_rows"] for report in reports),
            "rejected_rows": sum(report["rejected_rows"] for report in reports),
        },
        "reports": reports,
    }
    qc = verify_outputs(out_root, reports)
    if not qc["passed"]:
        raise SystemExit("QC failed: " + json.dumps(qc, ensure_ascii=False))
    print(json.dumps({"totals": summary["totals"], "qc_passed": qc["passed"]}, sort_keys=True))


# ------------------------------------------------------------------ batch 2

import argparse
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = pools.source_dir("pedagogical_action")
SOURCE_ROOT = DEFAULT_ROOT / "huggingface"
DATASETS_b2 = ("SocraticMATH", "SocraticMATH-sol", "EduAdapt")
SOCRATIC_SPLITS = {"train": "train", "val": "validation", "test": "test"}


def normalize_text_b2(value: Any, *, optional: bool = False) -> str | None:
    if value is None:
        return None if optional else ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or (None if optional else "")


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


def source_file_entry(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(DEFAULT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def dedup_key(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    return canonical_json({field: record.get(field) for field in fields})


def clean_socratic(dataset: str, out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = SOURCE_ROOT / dataset
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    input_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    source_files: list[dict[str, Any]] = []
    source_ids: dict[str, set[int]] = defaultdict(set)
    role_counts: Counter[str] = Counter()
    turn_counts: list[int] = []

    for file_split, split in SOCRATIC_SPLITS.items():
        paths = sorted((source_dir / "data").glob(f"{file_split}-*.parquet"))
        if len(paths) != 1:
            raise ValueError(f"{dataset}: expected one parquet for {file_split}, got {paths}")
        source = paths[0]
        source_files.append(source_file_entry(source))
        table = pq.read_table(source)
        expected = {"id", "conversations"}
        if set(table.column_names) != expected:
            raise ValueError(f"{dataset}: schema mismatch in {source}: {table.column_names}")

        for source_row_number, raw in enumerate(table.to_pylist(), 1):
            input_counts[split] += 1
            source_id = raw.get("id")
            conversations = raw.get("conversations")
            locator = {
                "source_file": str(source.relative_to(DEFAULT_ROOT)),
                "source_row_number": source_row_number,
                "source_record_id": source_id,
                "split": split,
            }
            if not isinstance(source_id, int):
                rejected.append({"reason": "invalid_source_record_id", **locator})
                continue
            if source_id in source_ids[split]:
                rejected.append({"reason": "duplicate_source_id_within_split", **locator})
                continue
            source_ids[split].add(source_id)
            if not isinstance(conversations, list) or not conversations:
                rejected.append({"reason": "empty_or_invalid_conversation", **locator})
                continue

            normalized_turns: list[dict[str, str]] = []
            bad_reason: str | None = None
            for turn in conversations:
                if not isinstance(turn, dict) or set(turn) != {"from", "value"}:
                    bad_reason = "invalid_turn_schema"
                    break
                role = normalize_text_b2(turn.get("from"))
                value = normalize_text_b2(turn.get("value"))
                if role not in {"user", "assistant"}:
                    bad_reason = "invalid_turn_role"
                    break
                if not value:
                    bad_reason = "empty_turn_value"
                    break
                normalized_turns.append({"role": role, "content": value})
            if bad_reason:
                rejected.append({"reason": bad_reason, **locator})
                continue

            record = {
                "dataset": dataset,
                "split": split,
                "source_record_id": source_id,
                "messages": normalized_turns,
                "provenance": {
                    "source_repository": f"ulises-c/{dataset}",
                    "original_creator": "ECNU-ICALK",
                    "original_project": "SocraticMath",
                    "source_kind": "manually_annotated_socratic_dialogue",
                    "synthetic_model": None,
                    "license": "CC BY-NC 4.0",
                    "variant": (
                        "solution_prepended_source_variant"
                        if dataset.endswith("-sol")
                        else "base_dialogue_variant"
                    ),
                },
            }
            key = dedup_key(record, ("split", "messages"))
            if key in seen:
                rejected.append(
                    {
                        "reason": "exact_normalized_duplicate",
                        "duplicate_of": seen[key],
                        **locator,
                    }
                )
                continue
            record_id = f"{dataset}:{split}:{source_id}"
            seen[key] = record_id
            records.append(record)
            kept_counts[split] += 1
            turn_counts.append(len(normalized_turns))
            role_counts.update(turn["role"] for turn in normalized_turns)

    write_jsonl(out_dir / "kept.jsonl", records)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = {
        "dataset": dataset,
        "scope": "deterministic_quick_clean_no_answer_rewrite_no_generation",
        "source_files": source_files,
        "input_rows": sum(input_counts.values()),
        "input_split_counts": dict(sorted(input_counts.items())),
        "kept_rows": len(records),
        "kept_split_counts": dict(sorted(kept_counts.items())),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "normalization": [
            "rename turn fields from/from,value to role,content",
            "CRLF/CR to LF and outer whitespace trim",
            "retain every utterance verbatim apart from line-ending/outer-whitespace normalization",
        ],
        "dedup_policy": "Exact equality of normalized split plus complete message sequence.",
        "split_policy": "Official train/validation/test split preserved; no split invented or moved.",
        "schema_checks": {
            "source_columns_exact": True,
            "source_ids_unique_within_each_split": True,
            "kept_roles_limited_to_user_assistant": True,
            "kept_nonempty_messages": True,
            "turn_count_min": min(turn_counts) if turn_counts else None,
            "turn_count_max": max(turn_counts) if turn_counts else None,
            "role_counts": dict(sorted(role_counts.items())),
        },
        "provenance_audit": {
            "audit_status": "documented",
            "original_creator": "ECNU-ICALK",
            "huggingface_uploader": "Ulises Chavarria",
            "content_origin_claim": "real Chinese primary-school exam questions with manually annotated Socratic dialogues",
            "synthetic_model": None,
            "license": "CC BY-NC 4.0",
            "evidence_files": [
                str((source_dir / "README.md").relative_to(DEFAULT_ROOT)),
                str((source_dir / "ATTRIBUTION.md").relative_to(DEFAULT_ROOT)),
            ],
            "commercial_use_warning": "Non-commercial license.",
        },
    }
    write_json(out_dir / "report.json", report)
    return report, records


def file_grade_subject(path: Path) -> tuple[str, str]:
    match = re.fullmatch(
        r"Grade_(.+)_(biology|chemistry|computer_science|ecology|geography|geology|medicine|metrology|physics)\.jsonl",
        path.name,
    )
    if not match:
        raise ValueError(f"Unexpected EduAdapt filename: {path.name}")
    grade_token, subject = match.groups()
    grade_level = "grade " + grade_token.replace("_and_", " and ").replace("_to_", " to ")
    return grade_level, subject


def resolve_mcq_answer(
    question: dict[str, Any], classification: dict[str, Any] | None
) -> tuple[str | None, str | None]:
    options = question.get("options")
    if not isinstance(options, list) or not options:
        return None, "invalid_mcq_options"
    normalized_options = [normalize_text_b2(item) for item in options]
    if any(not item for item in normalized_options):
        return None, "empty_mcq_option"
    raw = normalize_text_b2(
        question.get("correct_answer", question.get("correct_option")), optional=True
    )
    if not raw:
        return None, "missing_mcq_answer"
    if raw in normalized_options:
        return raw, None
    if len(raw) == 1 and raw.upper() in string.ascii_uppercase:
        index = ord(raw.upper()) - ord("A")
        if index < len(normalized_options):
            return normalized_options[index], None
        return None, "mcq_option_index_out_of_range"
    class_answer = (
        normalize_text_b2(classification.get("answer"), optional=True)
        if isinstance(classification, dict)
        else None
    )
    if class_answer in normalized_options:
        return class_answer, None
    return None, "mcq_answer_not_in_options"


def iter_eduadapt_candidates(
    path: Path,
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    grade_from_file, subject = file_grade_subject(path)
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                yield {}, {
                    "reason": "invalid_json",
                    "source_file": str(path.relative_to(DEFAULT_ROOT)),
                    "source_line": source_line,
                    "detail": str(exc),
                }
                continue
            locator = {
                "source_file": str(path.relative_to(DEFAULT_ROOT)),
                "source_line": source_line,
                "source_topic_id": raw.get("id"),
            }
            if not isinstance(raw, dict):
                yield {}, {"reason": "top_level_not_object", **locator}
                continue
            title = normalize_text_b2(raw.get("title"))
            if not title:
                yield {}, {"reason": "missing_title", **locator}
                continue

            if "questions" in raw:
                grade_level = normalize_text_b2(raw.get("grade_level"))
                if grade_level != grade_from_file:
                    yield {}, {
                        "reason": "grade_level_filename_mismatch",
                        "observed_grade_level": grade_level,
                        "expected_grade_level": grade_from_file,
                        **locator,
                    }
                    continue
                blocks = [
                    {
                        "questions": raw.get("questions"),
                        "article": None,
                        "ratings": None,
                        "rationales": None,
                        "classification": None,
                        "model_name": None,
                        "source_structure": "direct_questions",
                    }
                ]
            elif "evaluation" in raw:
                evaluations = raw.get("evaluation")
                if not isinstance(evaluations, list) or not evaluations:
                    yield {}, {"reason": "empty_or_invalid_evaluation", **locator}
                    continue
                blocks = []
                for evaluation_index, evaluation in enumerate(evaluations):
                    if not isinstance(evaluation, dict):
                        yield {}, {
                            "reason": "evaluation_not_object",
                            "evaluation_index": evaluation_index,
                            **locator,
                        }
                        continue
                    block = dict(evaluation)
                    block["source_structure"] = "evaluation_nested"
                    block["evaluation_index"] = evaluation_index
                    blocks.append(block)
            else:
                yield {}, {"reason": "unrecognized_top_level_schema", **locator}
                continue

            for block in blocks:
                grade_level = normalize_text_b2(block.get("grade_level", grade_from_file))
                if grade_level != grade_from_file:
                    yield {}, {
                        "reason": "grade_level_filename_mismatch",
                        "observed_grade_level": grade_level,
                        "expected_grade_level": grade_from_file,
                        **locator,
                    }
                    continue
                questions = block.get("questions")
                if not isinstance(questions, list) or not questions:
                    yield {}, {"reason": "empty_or_invalid_questions", **locator}
                    continue
                classifications = block.get("classification")
                ratings = block.get("ratings")
                rationales = block.get("rationales")
                if block["source_structure"] == "evaluation_nested":
                    lengths = {
                        "classification": (
                            len(classifications) if isinstance(classifications, list) else -1
                        ),
                        "ratings": len(ratings) if isinstance(ratings, list) else -1,
                        "rationales": len(rationales) if isinstance(rationales, list) else -1,
                    }
                    if any(length != len(questions) for length in lengths.values()):
                        yield {}, {
                            "reason": "evaluation_parallel_array_length_mismatch",
                            "question_count": len(questions),
                            "observed_lengths": lengths,
                            **locator,
                        }
                        continue

                for question_index, question in enumerate(questions):
                    qlocator = {
                        **locator,
                        "question_index": question_index,
                        "source_structure": block["source_structure"],
                    }
                    if not isinstance(question, dict):
                        yield {}, {"reason": "question_not_object", **qlocator}
                        continue
                    question_type = normalize_text_b2(question.get("type"))
                    question_text = normalize_text_b2(question.get("question"))
                    if not question_text:
                        yield {}, {"reason": "missing_question_text", **qlocator}
                        continue
                    classification = (
                        classifications[question_index]
                        if isinstance(classifications, list)
                        else None
                    )
                    if isinstance(classification, dict):
                        class_question = normalize_text_b2(
                            classification.get("question"), optional=True
                        )
                        if class_question and class_question != question_text:
                            yield {}, {
                                "reason": "classification_question_mismatch",
                                **qlocator,
                            }
                            continue

                    if question_type == "qa":
                        answer = normalize_text_b2(question.get("answer"))
                        options = None
                        if not answer:
                            yield {}, {"reason": "missing_answer", **qlocator}
                            continue
                    elif question_type == "mcq":
                        options = [normalize_text_b2(item) for item in question.get("options", [])]
                        answer, error = resolve_mcq_answer(question, classification)
                        if error:
                            yield {}, {"reason": error, **qlocator}
                            continue
                    else:
                        yield {}, {
                            "reason": "unsupported_question_type",
                            "observed_type": question_type,
                            **qlocator,
                        }
                        continue

                    article = normalize_text_b2(block.get("article"), optional=True)
                    model_name_raw = normalize_text_b2(block.get("model_name"), optional=True)
                    synthetic_model = Path(model_name_raw).name.lower() if model_name_raw else None
                    record = {
                        "dataset": "EduAdapt",
                        "split": "unsplit",
                        "source_record_id": stable_id(
                            "eduadapt",
                            [
                                str(path.relative_to(DEFAULT_ROOT)),
                                source_line,
                                block.get("evaluation_index"),
                                question_index,
                            ],
                        ),
                        "source_topic_id": raw.get("id"),
                        "title": title,
                        "grade_level": grade_level,
                        "subject": subject,
                        "question_type": question_type,
                        "question": question_text,
                        "options": options,
                        "answer": answer,
                        "context": article,
                        "provenance": {
                            "source_repository": "notefill/eduadapt",
                            "source_structure": block["source_structure"],
                            "dataset_curator": "Notefill",
                            "synthetic_model": synthetic_model,
                            "synthetic_model_raw": model_name_raw,
                            "license": "CC BY 4.0",
                            "source_data_origin": "not specified in local dataset card",
                            "rating": (
                                ratings[question_index] if isinstance(ratings, list) else None
                            ),
                            "rationale": (
                                normalize_text_b2(rationales[question_index], optional=True)
                                if isinstance(rationales, list)
                                else None
                            ),
                            "classification": classification,
                        },
                    }
                    yield record, qlocator


def clean_eduadapt(out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_dir = SOURCE_ROOT / "EduAdapt"
    paths = sorted(source_dir.glob("Grade_*.jsonl"))
    source_top_level_rows = 0
    source_question_instances = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source_top_level_rows += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(raw, dict):
                    if isinstance(raw.get("questions"), list):
                        source_question_instances += len(raw["questions"])
                    if isinstance(raw.get("evaluation"), list):
                        for evaluation in raw["evaluation"]:
                            if isinstance(evaluation, dict) and isinstance(
                                evaluation.get("questions"), list
                            ):
                                source_question_instances += len(evaluation["questions"])
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    candidates = 0
    structure_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    grade_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    direct_count = 0
    synthetic_count = 0

    for path in paths:
        for record, locator in iter_eduadapt_candidates(path):
            if not record:
                rejected.append(locator)
                continue
            candidates += 1
            key = dedup_key(
                record,
                ("question_type", "question", "options", "answer", "context"),
            )
            if key in seen:
                rejected.append(
                    {
                        "reason": "exact_normalized_duplicate",
                        "duplicate_of": seen[key],
                        **locator,
                    }
                )
                continue
            seen[key] = record["source_record_id"]
            records.append(record)
            provenance = record["provenance"]
            structure_counts[provenance["source_structure"]] += 1
            question_type_counts[record["question_type"]] += 1
            grade_counts[record["grade_level"]] += 1
            subject_counts[record["subject"]] += 1
            if provenance["synthetic_model"]:
                synthetic_count += 1
                model_counts[provenance["synthetic_model"]] += 1
            else:
                direct_count += 1

    write_jsonl(out_dir / "kept.jsonl", records)
    write_jsonl(out_dir / "rejected.jsonl", rejected)
    report = {
        "dataset": "EduAdapt",
        "scope": "deterministic_quick_clean_no_answer_rewrite_no_generation",
        "source_files": [source_file_entry(path) for path in paths],
        "source_top_level_rows": source_top_level_rows,
        "source_question_instances": source_question_instances,
        "normalized_qa_candidates": candidates,
        "kept_rows": len(records),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "split_counts": {"unsplit": len(records)},
        "split_policy": "Source has no official split; all records remain `unsplit`.",
        "normalization": [
            "flatten direct and evaluation-nested source structures to one QA/MCQ per record",
            "CRLF/CR to LF and outer whitespace trim",
            "map MCQ correct option letters to the existing option text",
            "normalize phi4 path spelling to synthetic_model=phi4 while preserving raw model path",
            "blank optional context/rationale/model fields to null",
            "retain existing question and answer wording; no content rewrite",
        ],
        "dedup_policy": "Exact equality of normalized type, question, options, answer, and context.",
        "kept_counts": {
            "source_structure": dict(sorted(structure_counts.items())),
            "question_type": dict(sorted(question_type_counts.items())),
            "grade_level": dict(sorted(grade_counts.items())),
            "subject": dict(sorted(subject_counts.items())),
        },
        "schema_checks": {
            "source_file_count": len(paths),
            "kept_answers_nonempty": True,
            "kept_questions_nonempty": True,
            "kept_mcq_answer_in_options": all(
                row["question_type"] != "mcq" or row["answer"] in row["options"] for row in records
            ),
            "split_preserved_as_unsplit": True,
        },
        "provenance_audit": {
            "audit_status": "partial_source_provenance_synthetic_model_identified",
            "dataset_curator": "Notefill",
            "license": "CC BY 4.0",
            "direct_question_records": direct_count,
            "evaluation_nested_records_with_model_provenance": synthetic_count,
            "synthetic_model_counts": dict(sorted(model_counts.items())),
            "synthetic_model_evidence_field": "evaluation[].model_name",
            "source_data_origin": "The local README says the dataset appears derived from educational content but does not name the underlying corpus.",
            "source_origin_warning": "Underlying article/source provenance is incomplete in the local dataset card.",
            "evidence_files": [str((source_dir / "README.md").relative_to(DEFAULT_ROOT))],
        },
    }
    write_json(out_dir / "report.json", report)
    return report, records


def audit_socratic_pair(
    base_records: list[dict[str, Any]], sol_records: list[dict[str, Any]]
) -> dict[str, Any]:
    base = {(r["split"], r["source_record_id"]): r for r in base_records}
    sol = {(r["split"], r["source_record_id"]): r for r in sol_records}
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    marker = "【解析】:"
    for key in sorted(set(base) | set(sol)):
        if key not in base:
            counts["missing_in_base"] += 1
            continue
        if key not in sol:
            counts["missing_in_sol"] += 1
            continue
        a = base[key]["messages"]
        b = sol[key]["messages"]
        if len(a) != len(b):
            counts["turn_count_mismatch"] += 1
            continue
        differences = [index for index, pair in enumerate(zip(a, b)) if pair[0] != pair[1]]
        first_expected = (
            differences == [0]
            and a[0]["role"] == b[0]["role"]
            and b[0]["content"].startswith(a[0]["content"])
            and marker in b[0]["content"][len(a[0]["content"]) :]
        )
        if first_expected:
            counts["expected_solution_append_only"] += 1
        elif not differences:
            counts["identical"] += 1
        else:
            counts["unexpected_difference"] += 1
            if len(examples) < 20:
                examples.append({"split": key[0], "source_record_id": key[1], "turns": differences})
    result = {
        "relationship": "paired source variants, not independent samples",
        "pair_key": ["split", "source_record_id"],
        "counts": dict(sorted(counts.items())),
        "all_pairs_expected_solution_append_only": counts
        == {"expected_solution_append_only": len(base)}
        and len(base) == len(sol),
        "dedup_training_warning": "Do not count base and -sol as independent examples; choose one variant or group by pair key.",
        "unexpected_examples": examples,
    }
    write_json(PIPELINE_DIR / "group2_socratic_pair_audit.json", result)
    return result


def run_qc(reports: dict[str, Any], pair_audit: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for dataset in DATASETS_b2:
        out_dir = PIPELINE_DIR / dataset / "cleaned"
        report = reports[dataset]
        records = [
            json.loads(line)
            for line in (out_dir / "kept.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        rejected = [
            json.loads(line)
            for line in (out_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        checks[f"{dataset}:records_count_matches_report"] = len(records) == report["kept_rows"]
        checks[f"{dataset}:rejected_count_matches_report"] = (
            len(rejected) == report["rejected_rows"]
        )
        checks[f"{dataset}:all_records_have_split"] = all(row.get("split") for row in records)
        checks[f"{dataset}:no_duplicate_output_lines"] = len(
            {canonical_json(row) for row in records}
        ) == len(records)
        if dataset.startswith("SocraticMATH"):
            checks[f"{dataset}:official_split_counts"] = report["kept_split_counts"] == {
                "test": 685,
                "train": 5476,
                "validation": 685,
            }
            checks[f"{dataset}:messages_valid"] = all(
                row.get("messages")
                and all(
                    turn.get("role") in {"user", "assistant"} and turn.get("content")
                    for turn in row["messages"]
                )
                for row in records
            )
        else:
            checks["EduAdapt:unsplit_only"] = {row["split"] for row in records} == {"unsplit"}
            checks["EduAdapt:mcq_answer_in_options"] = all(
                row["question_type"] != "mcq" or row["answer"] in row["options"] for row in records
            )
            checks["EduAdapt:no_empty_qa"] = all(
                row["question"] and row["answer"] for row in records
            )
    checks["Socratic_pair:expected_relationship"] = pair_audit[
        "all_pairs_expected_solution_append_only"
    ]
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "failed_checks": sorted(key for key, value in checks.items() if not value),
    }
    if result["status"] != "pass":
        raise RuntimeError(f"QC failed: {result['failed_checks']}")
    return result


def main_batch2() -> None:
    global DEFAULT_ROOT, SOURCE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    DEFAULT_ROOT = args.root.resolve()
    SOURCE_ROOT = DEFAULT_ROOT / "huggingface"

    reports: dict[str, Any] = {}
    reports["SocraticMATH"], base_records = clean_socratic(
        "SocraticMATH", PIPELINE_DIR / "SocraticMATH" / "cleaned"
    )
    reports["SocraticMATH-sol"], sol_records = clean_socratic(
        "SocraticMATH-sol", PIPELINE_DIR / "SocraticMATH-sol" / "cleaned"
    )
    reports["EduAdapt"], _ = clean_eduadapt(PIPELINE_DIR / "EduAdapt" / "cleaned")
    pair_audit = audit_socratic_pair(base_records, sol_records)
    qc = run_qc(reports, pair_audit)
    print(
        json.dumps(
            {"reports": reports, "pair_audit": pair_audit, "qc": qc}, ensure_ascii=False, indent=2
        )
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
