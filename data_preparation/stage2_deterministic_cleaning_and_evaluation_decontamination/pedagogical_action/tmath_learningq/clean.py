#!/usr/bin/env python3
"""TMATH and LearningQ cleaning with QC."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import sqlite3
import sys
import sys as _sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

SOURCE_ROOT = pools.source_dir("pedagogical_action")
TMATH_OUT = pools.cleaned_dir("pedagogical_action", "TMATH")
LEARNINGQ_OUT = pools.cleaned_dir("pedagogical_action", "LearningQ")
DEFAULT_TMATH = SOURCE_ROOT / "github" / "TMATH"
DEFAULT_LEARNINGQ = SOURCE_ROOT / "huggingface" / "LearningQ-qg"
TMATH_FIELDS = ("problem", "level", "type", "solution", "socratic_questions")
LEARNINGQ_FIELDS = ("context", "questionsrc", "question")
LEARNINGQ_SPLITS = ("validation", "test", "train")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_text(*values: str) -> str:
    h = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def source_entry(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
            count += 1
    return count


def is_blank(value: Any) -> bool:
    return not isinstance(value, str) or not value.strip()


def clean_tmath(source: Path, output: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    domains: collections.Counter[str] = collections.Counter()
    input_domains: collections.Counter[str] = collections.Counter()
    source_files = sorted(source.glob("hint_*/*.json"))

    for path in source_files:
        relative = path.relative_to(source)
        domain = path.parent.name.removeprefix("hint_")
        input_domains[domain] += 1
        locator = {"source_file": str(relative), "domain": domain}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            rejected.append({"reason": "invalid_json", "detail": str(exc), **locator})
            continue
        if not isinstance(raw, dict):
            rejected.append({"reason": "top_level_not_object", **locator})
            continue
        missing = [field for field in TMATH_FIELDS if field not in raw]
        extra = sorted(set(raw) - set(TMATH_FIELDS))
        if missing or extra:
            rejected.append(
                {
                    "reason": "bad_structure",
                    "missing_fields": missing,
                    "extra_fields": extra,
                    **locator,
                }
            )
            continue
        blank = [field for field in TMATH_FIELDS if is_blank(raw[field])]
        wrong_type = [field for field in TMATH_FIELDS if not isinstance(raw[field], str)]
        if blank or wrong_type:
            rejected.append(
                {
                    "reason": "empty_or_non_string_field",
                    "blank_fields": blank,
                    "non_string_fields": wrong_type,
                    **locator,
                }
            )
            continue

        content = {field: raw[field] for field in TMATH_FIELDS}
        key = digest_text(*(content[field] for field in TMATH_FIELDS))
        record_id = f"tmath:{domain}:{path.stem}"
        if key in seen:
            rejected.append(
                {
                    "reason": "exact_duplicate",
                    "duplicate_of": seen[key],
                    **locator,
                }
            )
            continue
        seen[key] = record_id
        records.append(
            {
                "id": record_id,
                "dataset": "TMATH",
                "domain": domain,
                **content,
                "provenance": {
                    "source_file": str(relative),
                    "official_split": None,
                    "license": None,
                    "license_status": "missing",
                },
            }
        )
        domains[domain] += 1

    write_jsonl(output / "kept.jsonl", records)
    write_jsonl(output / "rejected.jsonl", rejected)
    report = {
        "dataset": "TMATH",
        "scope": "deterministic_clean_no_text_rewrite",
        "source_root": str(source),
        "source_file_count": len(source_files),
        "source_readme": source_entry(source / "README.md"),
        "input_rows": len(source_files),
        "input_by_domain": dict(sorted(input_domains.items())),
        "kept_rows": len(records),
        "kept_by_domain": dict(sorted(domains.items())),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(
            sorted(collections.Counter(row["reason"] for row in rejected).items())
        ),
        "required_fields": list(TMATH_FIELDS),
        "dedup_policy": "Exact equality of all five decoded source text fields; first source path wins.",
        "text_policy": "No source text is stripped, normalized, split, joined, or rewritten.",
        "license": {
            "value": None,
            "status": "missing",
            "evidence": "No repository LICENSE file and no license declaration in README.md.",
        },
    }
    write_json(output / "report.json", report)
    return report


def open_csv_from_zip(zip_path: Path, member: str):
    archive = zipfile.ZipFile(zip_path)
    raw = archive.open(member, "r")
    text = __import__("io").TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    return archive, raw, text


def initialize_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=FILE;
        CREATE TABLE documents (
            split TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            context TEXT NOT NULL,
            first_source_row INTEGER NOT NULL,
            PRIMARY KEY (split, context_hash)
        );
        CREATE TABLE questions (
            split TEXT NOT NULL,
            context_hash TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            questionsrc TEXT NOT NULL,
            question TEXT NOT NULL,
            pair_hash TEXT NOT NULL,
            PRIMARY KEY (split, pair_hash)
        );
        CREATE INDEX questions_document_idx ON questions(split, context_hash, source_row);
        CREATE TABLE eval_pairs (
            pair_hash TEXT PRIMARY KEY
        );
        """
    )
    return db


def learningq_rows(source: Path, split: str):
    zip_path = source / f"{split}.zip"
    archive, raw, text = open_csv_from_zip(zip_path, f"{split}.csv")
    try:
        reader = csv.DictReader(text)
        if reader.fieldnames != list(LEARNINGQ_FIELDS):
            raise ValueError(
                f"{zip_path}: expected header {LEARNINGQ_FIELDS}, got {reader.fieldnames}"
            )
        for source_row, row in enumerate(reader, 2):
            yield source_row, row
    finally:
        text.close()
        raw.close()
        archive.close()


def grouped_documents(db: sqlite3.Connection, split: str):
    docs = db.execute(
        """
        SELECT context_hash, context, first_source_row
        FROM documents WHERE split=? ORDER BY first_source_row
        """,
        (split,),
    )
    for context_hash, context, first_source_row in docs:
        questions = [
            {
                "questionsrc": questionsrc,
                "question": question,
                "source_row": source_row,
            }
            for source_row, questionsrc, question in db.execute(
                """
                SELECT source_row, questionsrc, question
                FROM questions
                WHERE split=? AND context_hash=?
                ORDER BY source_row
                """,
                (split, context_hash),
            )
        ]
        yield {
            "id": f"learningq:{split}:{context_hash[:20]}",
            "dataset": "LearningQ-qg",
            "split": split,
            "context": context,
            "questions": questions,
            "provenance": {
                "source_archive": f"{split}.zip",
                "first_source_row": first_source_row,
                "license": "unknown",
                "license_status": "unknown",
            },
        }


def clean_learningq(source: Path, output: Path) -> dict[str, Any]:
    csv.field_size_limit(sys.maxsize)
    db_path = output / "learningq_work.sqlite3"
    db = initialize_db(db_path)
    counts: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in LEARNINGQ_SPLITS
    }
    rejected_path = output / "learningq_rejected.jsonl.gz"
    with gzip.open(rejected_path, "wt", encoding="utf-8", newline="\n") as rejected:
        for split in LEARNINGQ_SPLITS:
            for source_row, row in learningq_rows(source, split):
                counts[split]["input_rows"] += 1
                if None in row:
                    counts[split]["bad_column_count"] += 1
                    rejected.write(
                        canonical(
                            {
                                "dataset": "LearningQ-qg",
                                "split": split,
                                "source_row": source_row,
                                "reason": "bad_column_count",
                            }
                        )
                        + "\n"
                    )
                    continue
                blank = [field for field in LEARNINGQ_FIELDS if is_blank(row.get(field))]
                if blank:
                    counts[split]["empty_or_missing_field"] += 1
                    rejected.write(
                        canonical(
                            {
                                "dataset": "LearningQ-qg",
                                "split": split,
                                "source_row": source_row,
                                "reason": "empty_or_missing_field",
                                "fields": blank,
                            }
                        )
                        + "\n"
                    )
                    continue

                context, questionsrc, question = (
                    row["context"],
                    row["questionsrc"],
                    row["question"],
                )
                pair_hash = digest_text(context, questionsrc, question)
                context_hash = digest_text(context)
                if (
                    split == "train"
                    and db.execute(
                        "SELECT 1 FROM eval_pairs WHERE pair_hash=?", (pair_hash,)
                    ).fetchone()
                ):
                    counts[split]["exact_overlap_with_validation_or_test"] += 1
                    rejected.write(
                        canonical(
                            {
                                "dataset": "LearningQ-qg",
                                "split": split,
                                "source_row": source_row,
                                "reason": "exact_overlap_with_validation_or_test",
                            }
                        )
                        + "\n"
                    )
                    continue
                try:
                    db.execute(
                        """
                        INSERT INTO questions
                        (split, context_hash, source_row, questionsrc, question, pair_hash)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (split, context_hash, source_row, questionsrc, question, pair_hash),
                    )
                except sqlite3.IntegrityError:
                    counts[split]["exact_duplicate_within_split"] += 1
                    rejected.write(
                        canonical(
                            {
                                "dataset": "LearningQ-qg",
                                "split": split,
                                "source_row": source_row,
                                "reason": "exact_duplicate_within_split",
                            }
                        )
                        + "\n"
                    )
                    continue
                db.execute(
                    """
                    INSERT OR IGNORE INTO documents
                    (split, context_hash, context, first_source_row)
                    VALUES (?, ?, ?, ?)
                    """,
                    (split, context_hash, context, source_row),
                )
                if split != "train":
                    db.execute(
                        "INSERT OR IGNORE INTO eval_pairs(pair_hash) VALUES (?)",
                        (pair_hash,),
                    )
                counts[split]["kept_pairs"] += 1
                if counts[split]["input_rows"] % 10000 == 0:
                    db.commit()
            db.commit()

    output_names = {
        "train": "learningq_train_candidates.jsonl.gz",
        "validation": "learningq_validation_isolated.jsonl.gz",
        "test": "learningq_test_isolated.jsonl.gz",
    }
    for split in ("train", "validation", "test"):
        counts[split]["kept_documents"] = write_jsonl(
            output / output_names[split], grouped_documents(db, split)
        )

    source_files = [
        source_entry(source / f"{split}.zip") for split in ("train", "validation", "test")
    ]
    validation_hashes = {
        row[0] for row in db.execute("SELECT pair_hash FROM questions WHERE split='validation'")
    }
    test_hashes = {
        row[0] for row in db.execute("SELECT pair_hash FROM questions WHERE split='test'")
    }
    report = {
        "dataset": "LearningQ-qg",
        "scope": "deterministic_clean_no_text_rewrite",
        "source_root": str(source),
        "source_files": source_files,
        "counts": {split: dict(sorted(counts[split].items())) for split in counts},
        "outputs": output_names,
        "structure": {
            "unit": "exact document context",
            "questions": "ordered list of exact questionsrc/question pairs with source row",
        },
        "dedup_policy": (
            "Exact equality of decoded context, questionsrc, and question. "
            "Within-split duplicates are removed. Exact validation/test pairs are "
            "excluded from train candidates."
        ),
        "split_policy": (
            "Only official train rows can enter train candidates. Official validation "
            "and test are emitted separately and never moved into train."
        ),
        "validation_test_exact_overlap_pairs": len(validation_hashes & test_hashes),
        "text_policy": "No source text is stripped, normalized, joined, or rewritten.",
        "license": {
            "value": "unknown",
            "status": "unknown",
            "evidence": "Local dataset card declares license: unknown and gives no licensing terms.",
        },
    }
    write_json(output / "report.json", report)
    db.close()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(db_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    return report


def clean(tmath_path: Path, learningq_path: Path) -> dict[str, Any]:
    tmath = clean_tmath(tmath_path, TMATH_OUT)
    learningq = clean_learningq(learningq_path, LEARNINGQ_OUT)
    return {
        "TMATH": {
            "input_rows": tmath["input_rows"],
            "kept_rows": tmath["kept_rows"],
            "rejected_rows": tmath["rejected_rows"],
        },
        "LearningQ": {"counts": learningq["counts"]},
    }


# ---------------------------------------------------------------- qc

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

TMATH_FIELDS_qc = ("problem", "level", "type", "solution", "socratic_questions")


def digest_text_qc(*values: str) -> str:
    h = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)
    return h.hexdigest()


def iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            yield line_no, json.loads(line)


def qc(tmath_path: Path, learningq_path: Path) -> dict[str, Any]:
    assertions: dict[str, bool] = {}
    errors: list[dict[str, Any]] = []

    tmath_hashes: set[str] = set()
    tmath_domains: Counter[str] = Counter()
    tmath_count = 0
    for line_no, row in iter_jsonl(TMATH_OUT / "kept.jsonl"):
        tmath_count += 1
        if any(
            not isinstance(row.get(field), str) or not row[field].strip() for field in TMATH_FIELDS_qc
        ):
            errors.append(
                {"file": "TMATH/cleaned/kept.jsonl", "line": line_no, "error": "invalid_field"}
            )
        key = digest_text_qc(*(row[field] for field in TMATH_FIELDS_qc))
        if key in tmath_hashes:
            errors.append({"file": "TMATH/cleaned/kept.jsonl", "line": line_no, "error": "duplicate"})
        tmath_hashes.add(key)
        tmath_domains[row.get("domain", "")] += 1
        if row.get("provenance", {}).get("license_status") != "missing":
            errors.append(
                {"file": "TMATH/cleaned/kept.jsonl", "line": line_no, "error": "license_marker"}
            )

    split_files = {
        "train": "learningq_train_candidates.jsonl.gz",
        "validation": "learningq_validation_isolated.jsonl.gz",
        "test": "learningq_test_isolated.jsonl.gz",
    }
    pair_sets: dict[str, set[str]] = {}
    learningq_counts: dict[str, dict[str, int]] = {}
    for split, filename in split_files.items():
        pairs: set[str] = set()
        contexts: set[str] = set()
        documents = pair_count = 0
        previous_source_row = 0
        for line_no, row in iter_jsonl(LEARNINGQ_OUT / filename):
            documents += 1
            if row.get("split") != split:
                errors.append({"file": filename, "line": line_no, "error": "wrong_split"})
            context = row.get("context")
            if not isinstance(context, str) or not context.strip() or context in contexts:
                errors.append(
                    {"file": filename, "line": line_no, "error": "bad_or_duplicate_context"}
                )
            contexts.add(context)
            first_source_row = row.get("provenance", {}).get("first_source_row", 0)
            if first_source_row <= previous_source_row:
                errors.append({"file": filename, "line": line_no, "error": "document_order"})
            previous_source_row = first_source_row
            if row.get("provenance", {}).get("license_status") != "unknown":
                errors.append({"file": filename, "line": line_no, "error": "license_marker"})
            questions = row.get("questions")
            if not isinstance(questions, list) or not questions:
                errors.append({"file": filename, "line": line_no, "error": "empty_questions"})
                continue
            last_question_row = 0
            for question in questions:
                questionsrc = question.get("questionsrc")
                text = question.get("question")
                source_row = question.get("source_row", 0)
                if (
                    not isinstance(questionsrc, str)
                    or not questionsrc.strip()
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    errors.append({"file": filename, "line": line_no, "error": "invalid_question"})
                if source_row <= last_question_row:
                    errors.append({"file": filename, "line": line_no, "error": "question_order"})
                last_question_row = source_row
                key = digest_text_qc(context, questionsrc, text)
                if key in pairs:
                    errors.append({"file": filename, "line": line_no, "error": "duplicate_pair"})
                pairs.add(key)
                pair_count += 1
        pair_sets[split] = pairs
        learningq_counts[split] = {"documents": documents, "pairs": pair_count}

    assertions["tmath_nonempty"] = tmath_count > 0
    assertions["tmath_exact_dedup"] = len(tmath_hashes) == tmath_count
    assertions["tmath_all_domains_present"] = len(tmath_domains) == 7
    assertions["learningq_all_splits_nonempty"] = all(
        x["pairs"] > 0 for x in learningq_counts.values()
    )
    assertions["learningq_train_validation_isolated"] = not (
        pair_sets["train"] & pair_sets["validation"]
    )
    assertions["learningq_train_test_isolated"] = not (pair_sets["train"] & pair_sets["test"])

    source_tmath_hashes: set[str] = set()
    for path in sorted(tmath_path.glob("hint_*/*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        source_tmath_hashes.add(digest_text_qc(*(raw[field] for field in TMATH_FIELDS_qc)))
    assertions["tmath_text_source_fidelity"] = tmath_hashes <= source_tmath_hashes

    csv.field_size_limit(sys.maxsize)
    source_learningq_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        source_hashes: set[str] = set()
        input_rows = 0
        with zipfile.ZipFile(learningq_path / f"{split}.zip") as archive:
            with archive.open(f"{split}.csv") as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text)
                    for row in reader:
                        input_rows += 1
                        if None not in row and all(
                            isinstance(row.get(field), str) and row[field].strip()
                            for field in ("context", "questionsrc", "question")
                        ):
                            source_hashes.add(
                                digest_text_qc(row["context"], row["questionsrc"], row["question"])
                            )
        source_learningq_counts[split] = input_rows
        assertions[f"learningq_{split}_text_source_fidelity"] = pair_sets[split] <= source_hashes
    assertions["learningq_official_source_row_counts"] = source_learningq_counts == {
        "train": 188660,
        "validation": 20630,
        "test": 18227,
    }
    assertions["no_qc_errors"] = not errors
    report = {
        "status": "pass" if all(assertions.values()) else "fail",
        "assertions": assertions,
        "errors": errors[:100],
        "error_count": len(errors),
        "tmath": {"records": tmath_count, "domains": dict(sorted(tmath_domains.items()))},
        "learningq": learningq_counts,
        "learningq_source_rows": source_learningq_counts,
        "validation_test_exact_overlap_pairs": len(pair_sets["validation"] & pair_sets["test"]),
    }
    if report["status"] != "pass":
        raise SystemExit(json.dumps(report, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmath", type=Path, default=DEFAULT_TMATH)
    parser.add_argument("--learningq", type=Path, default=DEFAULT_LEARNINGQ)
    args = parser.parse_args()
    cleaned = clean(args.tmath, args.learningq)
    result = qc(args.tmath, args.learningq)
    print(json.dumps({"cleaned": cleaned, "assertions": result["assertions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
