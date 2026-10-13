#!/usr/bin/env python3
"""Deterministic, no-rewrite cleaning for FEAT, FOXGLOVE, and SEFORA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys as _sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FEAT_DIR = pools.cleaned_dir("pedagogical_action", "FEAT")
FOXGLOVE_DIR = pools.cleaned_dir("pedagogical_action", "FOXGLOVE")
SEFORA_DIR = pools.cleaned_dir("pedagogical_action", "SEFORA")
SOURCE_REPOS = {
    "FEAT": ROOT / "github" / "FEAT",
    "FOXGLOVE": ROOT / "github" / "FOXGLOVE",
    "SEFORA": ROOT / "github" / "SEFORA",
}
SEFORA_NAME = re.compile(r"Batch_(\d+)_Course_(\d+)_Class_(\d+)_Essay_(\d+)_(.+)_ID(\d+)\.json$")


def clean_string(value: Any) -> str:
    """Only trim boundary whitespace and normalize line endings."""
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any, n: int = 20) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:n]


def record_id(dataset: str, value: Any) -> str:
    return f"{dataset.lower()}_{digest(value)}"


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
            handle.write(canonical(row) + "\n")
            count += 1
    return count


def provenance(
    *,
    source_file: Path,
    source_row: int | None,
    source_type: str,
    feedback_origin: str | None = None,
    feedback_creator: str | None = None,
) -> dict[str, Any]:
    return {
        "repository": source_file.relative_to(ROOT).parts[1],
        "source_path": str(source_file.relative_to(ROOT)),
        "source_row": source_row,
        "source_type": source_type,
        "feedback_origin": feedback_origin,
        "feedback_creator": feedback_creator,
        "transformation": "schema_projection_and_boundary_whitespace_only_no_rewrite",
    }


def flags(
    *,
    authentic_student_writing: bool,
    deidentified: bool | None,
    source_warning: list[str],
    detected: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "privacy": {
            "authentic_student_writing": authentic_student_writing,
            "source_reports_deidentified": deidentified,
            "review_required": authentic_student_writing,
            "notes": (
                "Preserve source text; do not attempt regex redaction because it may "
                "alter student writing or create false positives."
                if authentic_student_writing
                else None
            ),
        },
        "safety": {
            "source_content_warnings": source_warning,
            "detected_conservative_markers": sorted(set(detected or [])),
            "review_required": bool(source_warning or detected),
            "text_modified": False,
        },
    }


def conservative_markers(text: str) -> list[str]:
    """Flags only; never filters or changes text."""
    patterns = {
        "strong_language": r"\b(fuck(?:ing|ed|er|s)?|shit(?:ty|s)?|bitch(?:es|y)?|asshole|damn)\b",
        "self_harm_or_suicide": r"\b(suicid(?:e|al)|kill myself|self[- ]harm)\b",
        "violence_or_threat": r"\b(kill(?:ed|ing)?|murder(?:ed|ing)?|shoot(?:ing|er)?|stab(?:bed|bing)?)\b",
        "sexual_content": r"\b(sex(?:ual|ually)?|rape(?:d)?|porn(?:ography|ographic)?)\b",
    }
    return [name for name, pat in patterns.items() if re.search(pat, text, re.I)]


def source_commit(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo / ".git" / value[5:]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        packed = repo / ".git" / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#") and line.endswith(" " + value[5:]):
                    return line.split()[0]
        return None
    return value


def clean_feat(out: Path) -> dict[str, Any]:
    source_root = SOURCE_REPOS["FEAT"] / "datasets" / "DIRECT-G"
    files = sorted(source_root.glob("*/*.json"))
    expected = {
        ("base", "train", 2): 3996,
        ("base", "train", 5): 3996,
        ("base", "test", 2): 444,
        ("base", "test", 5): 444,
        ("mixed", "train", 2): 7992,
        ("mixed", "train", 5): 7992,
        ("mixed", "test", 2): 888,
        ("mixed", "test", 5): 888,
    }
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    exact_seen: dict[str, str] = {}
    semantic_seen: dict[str, str] = {}
    semantic_splits: dict[str, set[str]] = defaultdict(set)
    exact_cross_versions = 0
    same_prompt_chosen_cross_versions = 0
    config_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for source in files:
        variant = source.parent.name
        match = re.fullmatch(r"(train|test)\.criteria_(2|5)\.json", source.name)
        if not match:
            raise ValueError(f"Unexpected FEAT file: {source}")
        split, criteria_s = match.groups()
        criteria = int(criteria_s)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if len(payload) != expected[(variant, split, criteria)]:
            raise ValueError(f"Unexpected FEAT count in {source}: {len(payload)}")
        for index, raw in enumerate(payload):
            required = {
                "data_id",
                "reply_id",
                "reply_type",
                "model_name",
                "prompt",
                "chosen",
                "rejected",
            }
            if not isinstance(raw, dict) or set(raw) != required:
                rejected.append(
                    {
                        "reason": "source_schema_mismatch",
                        "source_path": str(source.relative_to(ROOT)),
                        "source_row": index,
                    }
                )
                continue
            prompt = clean_string(raw["prompt"])
            chosen = clean_string(raw["chosen"])
            rejected_text = clean_string(raw["rejected"])
            if not prompt or not chosen or not rejected_text:
                rejected.append(
                    {
                        "reason": "empty_required_text",
                        "source_path": str(source.relative_to(ROOT)),
                        "source_row": index,
                    }
                )
                continue
            rid = record_id("FEAT", [str(source.relative_to(ROOT)), index, raw])
            exact_key = digest([prompt, chosen, rejected_text], 64)
            semantic_key = digest([prompt, chosen], 64)
            semantic_splits[semantic_key].add(split)
            duplicate = {
                "exact_content_duplicate_of": exact_seen.get(exact_key),
                "same_prompt_chosen_duplicate_of": semantic_seen.get(semantic_key),
                "is_canonical_exact_content": exact_key not in exact_seen,
                "is_canonical_prompt_chosen": semantic_key not in semantic_seen,
            }
            if exact_key in exact_seen:
                exact_cross_versions += 1
            else:
                exact_seen[exact_key] = rid
            if semantic_key in semantic_seen:
                same_prompt_chosen_cross_versions += 1
            else:
                semantic_seen[semantic_key] = rid
            origin = "model_generated"
            row = {
                "record_id": rid,
                "dataset": "FEAT",
                "task": "feedback_preference",
                "official_split": split,
                "source_partition": {
                    "dataset_family": "DIRECT-G",
                    "variant": variant,
                    "criteria_count": criteria,
                    "is_official_test": split == "test",
                    "training_eligibility": "evaluation_only" if split == "test" else "train",
                },
                "prompt": prompt,
                "chosen_feedback": chosen,
                "rejected_feedback": rejected_text,
                "source_metadata": {
                    "data_id": clean_string(raw["data_id"]),
                    "reply_id": clean_string(raw["reply_id"]),
                    "reply_type": raw["reply_type"],
                    "model_name": clean_string(raw["model_name"]),
                },
                "provenance": provenance(
                    source_file=source,
                    source_row=index,
                    source_type="official_release",
                    feedback_origin=origin,
                    feedback_creator=clean_string(raw["model_name"]),
                ),
                "privacy_safety": flags(
                    authentic_student_writing=False,
                    deidentified=None,
                    source_warning=[],
                    detected=conservative_markers("\n".join([prompt, chosen, rejected_text])),
                ),
                "dedup": duplicate,
            }
            rows.append(row)
            config_counts[f"{variant}/{split}/criteria_{criteria}"] += 1
            split_counts[split] += 1

    write_jsonl(out / "all_versions.jsonl", rows)
    canonical_rows = [r for r in rows if r["dedup"]["is_canonical_prompt_chosen"]]
    write_jsonl(out / "kept.jsonl", canonical_rows)
    write_jsonl(out / "rejected.jsonl", rejected)
    report = {
        "dataset": "FEAT",
        "source_commit": source_commit(SOURCE_REPOS["FEAT"]),
        "input_rows": sum(expected.values()),
        "valid_rows_all_versions": len(rows),
        "canonical_rows_same_prompt_chosen": len(canonical_rows),
        "rejected_rows": len(rejected),
        "official_split_counts_all_versions": dict(sorted(split_counts.items())),
        "configuration_counts": dict(sorted(config_counts.items())),
        "exact_content_cross_version_duplicate_rows": exact_cross_versions,
        "same_prompt_chosen_cross_version_duplicate_rows": same_prompt_chosen_cross_versions,
        "prompt_chosen_groups_crossing_official_splits": sum(
            len(splits) > 1 for splits in semantic_splits.values()
        ),
        "canonical_split_counts": dict(
            sorted(Counter(row["official_split"] for row in canonical_rows).items())
        ),
        "policy": (
        "All official configurations are preserved in all_versions. "
        "kept.jsonl keeps the first deterministic occurrence of each exact "
        "(prompt, chosen) pair, preventing base/mixed and criteria-version stacking. "
            "Files are processed with test before train, so any exact prompt/chosen "
            "cross-split collision is retained only as evaluation_only and excluded "
            "from canonical training. Official test remains evaluation_only."
        ),
    }
    write_json(out / "report.json", report)
    return report


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_foxglove(out: Path) -> dict[str, Any]:
    repo = SOURCE_REPOS["FOXGLOVE"]
    essay_source = repo / "essay_goals.csv"
    feedback_source = repo / "feedback.csv"
    global_source = repo / "ratings_global.csv"
    item_source = repo / "ratings_items.csv"
    essays_raw = read_csv(essay_source)
    feedback_raw = read_csv(feedback_source)
    global_raw = read_csv(global_source)
    item_raw = read_csv(item_source)
    feedback_essay_ids = {
        clean_string(row["essay_id"]) for row in feedback_raw if clean_string(row["essay_id"])
    }
    essay_by_id: dict[str, str] = {}
    essays: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for i, raw in enumerate(essays_raw, 2):
        essay_id = clean_string(raw["essay_id"])
        text = clean_string(raw[" Essay Contents"])
        goals = [clean_string(raw[f"goal_{j}"]) for j in range(1, 4)]
        if not essay_id or not text or any(not x for x in goals):
            rejected.append({"reason": "invalid_essay_row", "source_row": i})
            continue
        if essay_id in essay_by_id:
            rejected.append({"reason": "duplicate_essay_id", "source_row": i, "essay_id": essay_id})
            continue
        essay_by_id[essay_id] = text
        essays.append(
            {
                "record_id": record_id("FOXGLOVE_ESSAY", [essay_id, text, goals]),
                "dataset": "FOXGLOVE",
                "record_type": "essay",
                "essay_id": essay_id,
                "essay_text": text,
                "feedback_goals": goals,
                "official_split": "unsplit",
                "source_partition": {
                    "source_corpus": "PERSUADE_2.0",
                    "grade_level": "12",
                    "has_released_feedback": essay_id in feedback_essay_ids,
                    "training_eligibility": "unsplit_requires_downstream_split",
                },
                "provenance": provenance(
                    source_file=essay_source, source_row=i, source_type="official_release"
                ),
                "privacy_safety": flags(
                    authentic_student_writing=True,
                    deidentified=True,
                    source_warning=[],
                    detected=conservative_markers(text),
                ),
            }
        )

    feedback: list[dict[str, Any]] = []
    exact_seen: dict[str, str] = {}
    exact_dup_rows = 0
    source_counts: Counter[str] = Counter()
    for i, raw in enumerate(feedback_raw, 2):
        essay_id = clean_string(raw["essay_id"])
        content = clean_string(raw["feedback_content"])
        model = clean_string(raw["model"])
        if essay_id not in essay_by_id or not content or not model:
            rejected.append({"reason": "invalid_feedback_row", "source_row": i})
            continue
        origin = "teacher" if model == "human" else "model_generated"
        creator = clean_string(raw["reviewer_id"]) or model
        rid = record_id("FOXGLOVE_FEEDBACK", [i, raw])
        key = digest(
            [
                essay_id,
                model,
                creator,
                clean_string(raw["goal_addressed"]),
                clean_string(raw["urgency_rank"]),
                content,
                clean_string(raw["sentences_or_sections"]),
            ],
            64,
        )
        duplicate_of = exact_seen.get(key)
        if duplicate_of:
            exact_dup_rows += 1
        else:
            exact_seen[key] = rid
        row = {
            "record_id": rid,
            "dataset": "FOXGLOVE",
            "record_type": "feedback",
            "essay_id": essay_id,
            "official_split": "unsplit",
            "feedback": content,
            "anchor_text": clean_string(raw["sentences_or_sections"]) or None,
            "goal_addressed": clean_string(raw["goal_addressed"]),
            "urgency_rank": int(raw["urgency_rank"]) if clean_string(raw["urgency_rank"]) else None,
            "provenance": provenance(
                source_file=feedback_source,
                source_row=i,
                source_type="official_release",
                feedback_origin=origin,
                feedback_creator=creator,
            ),
            "source_metadata": {
                "model": model,
                "reviewer_id": clean_string(raw["reviewer_id"]) or None,
            },
            "privacy_safety": flags(
                authentic_student_writing=True,
                deidentified=True,
                source_warning=[],
                detected=conservative_markers("\n".join([essay_by_id[essay_id], content])),
            ),
            "dedup": {
                "exact_feedback_record_duplicate_of": duplicate_of,
                "is_canonical_exact_feedback_record": duplicate_of is None,
            },
        }
        feedback.append(row)
        source_counts[origin] += 1

    ratings: list[dict[str, Any]] = []
    rating_seen: dict[str, str] = {}
    rating_dup_rows = 0
    for source, kind, source_rows in [
        (global_source, "global", global_raw),
        (item_source, "item", item_raw),
    ]:
        for i, raw in enumerate(source_rows, 2):
            essay_id = clean_string(raw["essay_id"])
            if essay_id not in essay_by_id:
                rejected.append({"reason": "rating_missing_essay", "source_row": i})
                continue
            rating_values = {
                key: int(value)
                for key, value in raw.items()
                if key
                in {
                    "accuracy_content",
                    "actionability",
                    "clarity",
                    "relevance",
                    "specificity",
                    "tone",
                    "accuracy_position",
                }
                and clean_string(value)
            }
            rid = record_id("FOXGLOVE_RATING", [str(source.relative_to(ROOT)), i, raw])
            key = digest(raw, 64)
            duplicate_of = rating_seen.get(key)
            if duplicate_of:
                rating_dup_rows += 1
            else:
                rating_seen[key] = rid
            ratings.append(
                {
                    "record_id": rid,
                    "dataset": "FOXGLOVE",
                    "record_type": "expert_rating",
                    "rating_scope": kind,
                    "essay_id": essay_id,
                    "feedback_lookup": {
                        "real_model": clean_string(raw["real_model"]),
                        "reviewer_id": clean_string(raw["reviewer_id"]),
                        "anonymous_label": clean_string(raw["anonymous_label"]),
                        "feedback_id": clean_string(raw.get("feedback_id")) or None,
                    },
                    "rater_id": clean_string(raw["rater_id"]),
                    "ratings": rating_values,
                    "provenance": provenance(
                        source_file=source,
                        source_row=i,
                        source_type="expert_evaluation",
                        feedback_origin="expert_rating",
                        feedback_creator=clean_string(raw["rater_id"]),
                    ),
                    "dedup": {
                        "exact_rating_duplicate_of": duplicate_of,
                        "is_canonical_exact_rating": duplicate_of is None,
                    },
                }
            )

    write_jsonl(out / "essays.jsonl", essays)
    write_jsonl(out / "feedback.jsonl", feedback)
    write_jsonl(out / "ratings.jsonl", ratings)
    write_jsonl(out / "rejected.jsonl", rejected)
    report = {
        "dataset": "FOXGLOVE",
        "source_commit": source_commit(repo),
        "input": {
            "essay_rows": len(essays_raw),
            "feedback_rows": len(feedback_raw),
            "global_rating_rows": len(global_raw),
            "item_rating_rows": len(item_raw),
        },
        "output": {
            "essays": len(essays),
            "feedback": len(feedback),
            "ratings": len(ratings),
            "rejected": len(rejected),
        },
        "feedback_provenance_counts": dict(sorted(source_counts.items())),
        "essays_with_released_feedback": sum(
            row["source_partition"]["has_released_feedback"] for row in essays
        ),
        "essays_without_released_feedback": sum(
            not row["source_partition"]["has_released_feedback"] for row in essays
        ),
        "exact_feedback_record_duplicate_rows": exact_dup_rows,
        "exact_rating_duplicate_rows": rating_dup_rows,
        "official_split": "unsplit",
        "policy": (
            "Human instructor and model feedback remain explicitly separated by "
            "provenance. Expert ratings remain separate records and retain source "
            "lookup keys; no inferred join or text rewrite is performed."
        ),
    }
    write_json(out / "report.json", report)
    return report


def find_nearest_prompt(path: Path, data_root: Path) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    current = path.parent
    while current != data_root.parent:
        for txt in sorted(current.glob("*.txt")):
            results.append(
                {
                    "source_path": str(txt.relative_to(ROOT)),
                    "text": clean_string(txt.read_text(encoding="utf-8", errors="replace")),
                }
            )
        if results or current == data_root:
            break
        current = current.parent
    return results


def clean_sefora(out: Path) -> dict[str, Any]:
    repo = SOURCE_REPOS["SEFORA"]
    data_root = repo / "SEFORA"
    files = sorted(data_root.rglob("*.json"))
    documents: list[dict[str, Any]] = []
    feedback: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    essay_seen: dict[str, str] = {}
    feedback_seen_within_student_assignment: dict[str, str] = {}
    exact_essay_duplicates = 0
    cross_version_feedback_duplicates = 0
    empty_feedback = 0
    stages: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    feedback_scope_counts: Counter[str] = Counter()

    for source in files:
        match = SEFORA_NAME.fullmatch(source.name)
        if not match:
            rejected.append(
                {
                    "reason": "filename_schema_mismatch",
                    "source_path": str(source.relative_to(ROOT)),
                }
            )
            continue
        batch, course, class_id, assignment, stage, student_id = match.groups()
        raw = json.loads(source.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or "paragraphs" not in raw
            or not isinstance(raw["paragraphs"], list)
        ):
            rejected.append(
                {
                    "reason": "document_schema_mismatch",
                    "source_path": str(source.relative_to(ROOT)),
                }
            )
            continue
        paragraphs = []
        essay_parts = []
        for paragraph_index, paragraph in enumerate(raw["paragraphs"]):
            if not isinstance(paragraph, dict):
                continue
            body = clean_string(paragraph.get("body"))
            essay_parts.append(body)
            paragraphs.append(
                {
                    "paragraph_index": paragraph_index,
                    "body": body,
                }
            )
        essay_text = "\n\n".join(part for part in essay_parts if part)
        doc_id = record_id("SEFORA_DOCUMENT", [str(source.relative_to(ROOT)), raw])
        essay_key = digest(essay_text, 64)
        duplicate_of = essay_seen.get(essay_key) if essay_text else None
        if duplicate_of:
            exact_essay_duplicates += 1
        elif essay_text:
            essay_seen[essay_key] = doc_id
        markers = conservative_markers(essay_text)
        for marker in markers:
            marker_counts[marker] += 1
        documents.append(
            {
                "record_id": doc_id,
                "dataset": "SEFORA",
                "record_type": "student_essay_draft",
                "official_split": "unsplit",
                "document_identity": {
                    "batch": batch,
                    "course": course,
                    "class": class_id,
                    "assignment": assignment,
                    "stage": stage,
                    "student_pseudonym_id": student_id,
                    "student_assignment_key": (
                        f"Batch_{batch}/Course_{course}/Class_{class_id}/"
                        f"Essay_{assignment}/ID{student_id}"
                    ),
                },
                "paragraphs": paragraphs,
                "essay_text": essay_text,
                "assignment_materials": find_nearest_prompt(source, data_root),
                "provenance": provenance(
                    source_file=source,
                    source_row=None,
                    source_type="official_release_authentic_student_writing",
                ),
                "privacy_safety": flags(
                    authentic_student_writing=True,
                    deidentified=True,
                    source_warning=["vulgar_or_strong_language", "strong_opinions_or_emotion"],
                    detected=markers,
                ),
                "dedup": {
                    "exact_essay_duplicate_of": duplicate_of,
                    "is_canonical_exact_essay": duplicate_of is None,
                },
            }
        )
        stages[stage] += 1

        assignment_key = [batch, course, class_id, assignment, student_id]
        for paragraph_index, paragraph in enumerate(raw["paragraphs"]):
            if not isinstance(paragraph, dict):
                continue
            annotations = paragraph.get("annotations", [])
            if not isinstance(annotations, list):
                continue
            for annotation_index, annotation in enumerate(annotations):
                if not isinstance(annotation, dict):
                    continue
                comment = clean_string(annotation.get("comment"))
                if not comment:
                    empty_feedback += 1
                    rejected.append(
                        {
                            "reason": "empty_instructor_feedback",
                            "source_path": str(source.relative_to(ROOT)),
                            "paragraph_index": paragraph_index,
                            "annotation_index": annotation_index,
                        }
                    )
                    continue
                feedback_scope_counts["inline"] += 1
                fb_id = record_id(
                    "SEFORA_FEEDBACK",
                    [
                        str(source.relative_to(ROOT)),
                        "inline",
                        paragraph_index,
                        annotation_index,
                        annotation,
                    ],
                )
                fb_key = digest([assignment_key, comment], 64)
                fb_duplicate_of = feedback_seen_within_student_assignment.get(fb_key)
                if fb_duplicate_of:
                    cross_version_feedback_duplicates += 1
                else:
                    feedback_seen_within_student_assignment[fb_key] = fb_id
                feedback.append(
                    {
                        "record_id": fb_id,
                        "dataset": "SEFORA",
                        "record_type": "instructor_feedback",
                        "document_id": doc_id,
                        "student_assignment_key": documents[-1]["document_identity"][
                            "student_assignment_key"
                        ],
                        "stage": stage,
                        "feedback_scope": "inline",
                        "paragraph_index": paragraph_index,
                        "annotation_index": annotation_index,
                        "annotation_type": clean_string(annotation.get("type")),
                        "feedback": comment,
                        "anchor": {
                            "text": clean_string(annotation.get("text")) or None,
                            "context_left": clean_string(annotation.get("context_left")) or None,
                            "context_right": clean_string(annotation.get("context_right")) or None,
                            "color": clean_string(annotation.get("color")) or None,
                        },
                        "provenance": provenance(
                            source_file=source,
                            source_row=None,
                            source_type="official_release_instructor_annotation",
                            feedback_origin="teacher",
                            feedback_creator="course_instructor_anonymized",
                        ),
                        "privacy_safety": flags(
                            authentic_student_writing=True,
                            deidentified=True,
                            source_warning=[
                                "vulgar_or_strong_language",
                                "strong_opinions_or_emotion",
                            ],
                            detected=conservative_markers(comment),
                        ),
                        "dedup": {
                            "same_feedback_within_student_assignment_duplicate_of": fb_duplicate_of,
                            "is_canonical_within_student_assignment": fb_duplicate_of is None,
                        },
                    }
                )

        overall_annotations = raw.get("annotations", [])
        if isinstance(overall_annotations, list):
            for annotation_index, annotation in enumerate(overall_annotations):
                if not isinstance(annotation, dict):
                    continue
                comment = clean_string(annotation.get("comment"))
                grades = annotation.get("grades")
                if not comment and not grades:
                    empty_feedback += 1
                    rejected.append(
                        {
                            "reason": "empty_overall_instructor_feedback",
                            "source_path": str(source.relative_to(ROOT)),
                            "annotation_index": annotation_index,
                        }
                    )
                    continue
                feedback_scope_counts["overall"] += 1
                fb_id = record_id(
                    "SEFORA_FEEDBACK",
                    [
                        str(source.relative_to(ROOT)),
                        "overall",
                        annotation_index,
                        annotation,
                    ],
                )
                fb_key = digest([assignment_key, comment, grades], 64)
                fb_duplicate_of = feedback_seen_within_student_assignment.get(fb_key)
                if fb_duplicate_of:
                    cross_version_feedback_duplicates += 1
                else:
                    feedback_seen_within_student_assignment[fb_key] = fb_id
                feedback.append(
                    {
                        "record_id": fb_id,
                        "dataset": "SEFORA",
                        "record_type": "instructor_feedback",
                        "document_id": doc_id,
                        "student_assignment_key": documents[-1]["document_identity"][
                            "student_assignment_key"
                        ],
                        "stage": stage,
                        "feedback_scope": "overall",
                        "annotation_index": annotation_index,
                        "feedback": comment or None,
                        "grades": grades if isinstance(grades, dict) else None,
                        "provenance": provenance(
                            source_file=source,
                            source_row=None,
                            source_type="official_release_instructor_assessment",
                            feedback_origin="teacher",
                            feedback_creator="course_instructor_anonymized",
                        ),
                        "privacy_safety": flags(
                            authentic_student_writing=True,
                            deidentified=True,
                            source_warning=[
                                "vulgar_or_strong_language",
                                "strong_opinions_or_emotion",
                            ],
                            detected=conservative_markers(comment),
                        ),
                        "dedup": {
                            "same_feedback_within_student_assignment_duplicate_of": fb_duplicate_of,
                            "is_canonical_within_student_assignment": fb_duplicate_of is None,
                        },
                    }
                )

    write_jsonl(out / "documents_all_versions.jsonl", documents)
    write_jsonl(out / "documents.jsonl", [r for r in documents if r["dedup"]["is_canonical_exact_essay"]])
    write_jsonl(out / "feedback_all_versions.jsonl", feedback)
    write_jsonl(
        out / "feedback_canonical_within_student_assignment.jsonl",
        [r for r in feedback if r["dedup"]["is_canonical_within_student_assignment"]],
    )
    write_jsonl(out / "rejected.jsonl", rejected)
    report = {
        "dataset": "SEFORA",
        "source_commit": source_commit(repo),
        "input_json_documents": len(files),
        "output_documents_all_versions": len(documents),
        "output_documents_canonical_exact": sum(
            r["dedup"]["is_canonical_exact_essay"] for r in documents
        ),
        "output_feedback_all_versions": len(feedback),
        "output_feedback_canonical_within_student_assignment": sum(
            r["dedup"]["is_canonical_within_student_assignment"] for r in feedback
        ),
        "feedback_scope_counts": dict(sorted(feedback_scope_counts.items())),
        "stage_counts": dict(sorted(stages.items())),
        "exact_essay_cross_version_duplicate_rows": exact_essay_duplicates,
        "same_feedback_within_student_assignment_cross_version_duplicate_rows": (
            cross_version_feedback_duplicates
        ),
        "empty_feedback_rejected": empty_feedback,
        "rejected_rows": len(rejected),
        "documents_with_detected_safety_markers": dict(sorted(marker_counts.items())),
        "policy": (
            "All stages remain available in all_versions. Canonical files remove only "
            "exact essay duplicates globally and exact repeated feedback within the "
            "same pseudonymous student+assignment across stages. No student or teacher "
            "text is rewritten. Authentic-writing privacy and source content warnings "
            "are attached to every applicable record."
        ),
    }
    write_json(out / "report.json", report)
    return report


def qc(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    paths = [
        FEAT_DIR / "all_versions.jsonl",
        FEAT_DIR / "kept.jsonl",
        FOXGLOVE_DIR / "essays.jsonl",
        FOXGLOVE_DIR / "feedback.jsonl",
        FOXGLOVE_DIR / "ratings.jsonl",
        SEFORA_DIR / "documents_all_versions.jsonl",
        SEFORA_DIR / "feedback_all_versions.jsonl",
    ]
    for path in paths:
        count = 0
        ids: set[str] = set()
        valid = True
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    count += 1
                    if "record_id" not in row or row["record_id"] in ids:
                        valid = False
                    ids.add(row.get("record_id", ""))
                except Exception:
                    valid = False
        add(f"jsonl_valid_unique_ids:{path.relative_to(pools.STAGE2)}", valid, count)

    add(
        "feat_expected_all_versions",
        reports["FEAT"]["valid_rows_all_versions"] == 26640,
        reports["FEAT"]["valid_rows_all_versions"],
    )
    add(
        "feat_expected_canonical",
        reports["FEAT"]["canonical_rows_same_prompt_chosen"] == 8872,
        reports["FEAT"]["canonical_rows_same_prompt_chosen"],
    )
    add(
        "foxglove_expected_feedback",
        reports["FOXGLOVE"]["output"]["feedback"] == 2340,
        reports["FOXGLOVE"]["output"]["feedback"],
    )
    add(
        "foxglove_teacher_model_split",
        reports["FOXGLOVE"]["feedback_provenance_counts"]
        == {"model_generated": 1644, "teacher": 696},
        reports["FOXGLOVE"]["feedback_provenance_counts"],
    )
    add(
        "sefora_expected_documents",
        reports["SEFORA"]["output_documents_all_versions"] == 564,
        reports["SEFORA"]["output_documents_all_versions"],
    )
    add(
        "sefora_no_model_feedback",
        all(
            json.loads(line)["provenance"]["feedback_origin"] == "teacher"
            for line in (SEFORA_DIR / "feedback_all_versions.jsonl").open(encoding="utf-8")
        ),
        "all feedback provenance=teacher",
    )
    add(
        "sefora_privacy_flags_present",
        all(
            json.loads(line)["privacy_safety"]["privacy"]["review_required"]
            for line in (SEFORA_DIR / "documents_all_versions.jsonl").open(encoding="utf-8")
        ),
        "all documents review_required=true",
    )
    result = {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "reports": reports,
    }
    if not result["passed"]:
        raise RuntimeError("QC failed: " + json.dumps(result["checks"], ensure_ascii=False))
    return result


def load(path, key):
    d = {}
    for l in path.open():
        x = json.loads(l)
        d[str(x[key])] = x
    return d


def join_feedback_context() -> dict[str, int]:
    fox_docs = load(FOXGLOVE_DIR / "essays.jsonl", "essay_id")
    with (FOXGLOVE_DIR / "kept.jsonl").open("w") as out:
        for l in (FOXGLOVE_DIR / "feedback.jsonl").open():
            x = json.loads(l)
            doc = fox_docs.get(str(x["essay_id"]))
            x["essay_context"] = doc.get("essay_text") if doc else None
            out.write(json.dumps(x, ensure_ascii=False) + "\n")
    se_docs = {}
    for l in (SEFORA_DIR / "documents_all_versions.jsonl").open():
        x = json.loads(l)
        se_docs[x["record_id"]] = x
    with (SEFORA_DIR / "kept.jsonl").open("w") as out:
        for l in (SEFORA_DIR / "feedback_canonical_within_student_assignment.jsonl").open():
            x = json.loads(l)
            doc = se_docs.get(x["document_id"])
            x["essay_context"] = doc.get("essay_text") if doc else None
            if doc and x.get("paragraph_index") is not None:
                ps = doc.get("paragraphs") or []
                i = x["paragraph_index"]
                x["paragraph_context"] = ps[i] if isinstance(i, int) and 0 <= i < len(ps) else None
            out.write(json.dumps(x, ensure_ascii=False) + "\n")
    return {
        "FOXGLOVE": sum(1 for _ in (FOXGLOVE_DIR / "kept.jsonl").open()),
        "SEFORA": sum(1 for _ in (SEFORA_DIR / "kept.jsonl").open()),
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    reports = {
        "FEAT": clean_feat(FEAT_DIR),
        "FOXGLOVE": clean_foxglove(FOXGLOVE_DIR),
        "SEFORA": clean_sefora(SEFORA_DIR),
    }
    joined = join_feedback_context()
    result = qc(reports)
    print(json.dumps({"qc_passed": result["passed"], "kept": joined}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
