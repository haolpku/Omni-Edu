#!/usr/bin/env python3
"""Deterministic structural clean for SocraTeach and ConvoLearn.

No utterance is rewritten. The pipeline only normalizes line endings and outer
whitespace, maps source roles to user/assistant, removes empty/invalid rows or
turns, and performs exact normalized deduplication within each source variant.

SocraTeach single examples are derived turn-level augmentations of the same
underlying SocraTeach dialogues. They are emitted separately and linked to
their source problem/dialogue; aggregate counts therefore treat multi and
single as related variants, not independent dialogue corpora.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import cleaning, pools
from typing import Any, Iterable

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent.parent.parent.parent.parent
ACTION_ROOT = HERE.parents[1]
SOCRATEACH_DIR = ACTION_ROOT / "github" / "SocraticLM"
CONVOLEARN_DIR = ACTION_ROOT / "huggingface" / "ConvoLearn"
SINGLE_ID_RE = re.compile(
    r"^(irrelevant|question|incorrect|correct)#"
    r"((GSM8K|MAWPS)_(train|test)_\d+)_(\d+)_(\d+)@(.*)$"
)
CONVO_LINE_RE = re.compile(r"^\s*(Student|Teacher):\s*(.*?)\s*$")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


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


def source_file_entry(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(ACTION_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def parse_problem_id(problem_id: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"(GSM8K|MAWPS)_(train|test)_\d+", problem_id)
    return match.groups() if match else None


def normalize_pair_history(raw: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(raw, list) or not raw:
        return None, "empty_or_invalid_history"
    messages: list[dict[str, str]] = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            return None, "invalid_history_pair"
        student = normalize_text(pair[0])
        teacher = normalize_text(pair[1])
        if not student or not teacher:
            return None, "empty_history_message"
        messages.extend(
            [
                {"role": "user", "content": student},
                {"role": "assistant", "content": teacher},
            ]
        )
    return messages, None


def clean_socrateach(out_dir: Path) -> dict[str, Any]:
    multi_path = SOCRATEACH_DIR / "data" / "SocraTeach_multi.json"
    single_path = SOCRATEACH_DIR / "data" / "SocraTeach_single.json"
    multi_raw = json.loads(multi_path.read_text(encoding="utf-8"))
    single_raw = json.loads(single_path.read_text(encoding="utf-8"))

    multi_records: list[dict[str, Any]] = []
    single_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    multi_seen: dict[str, str] = {}
    single_seen: dict[str, str] = {}
    valid_dialogues: set[str] = set()
    multi_split_counts: Counter[str] = Counter()
    single_split_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    dialogue_turn_counts: list[int] = []
    empty_turns_dropped = 0

    for problem_id, problem in multi_raw.items():
        parsed = parse_problem_id(problem_id)
        if not parsed:
            rejected.append(
                {
                    "variant": "multi",
                    "source_record_id": problem_id,
                    "reason": "invalid_problem_id",
                }
            )
            continue
        benchmark, split = parsed
        if not isinstance(problem, dict) or set(problem) != {
            "question",
            "analysis",
            "answer",
            "steps",
            "dialogues",
        }:
            rejected.append(
                {
                    "variant": "multi",
                    "source_record_id": problem_id,
                    "reason": "invalid_problem_schema",
                }
            )
            continue
        dialogues = problem.get("dialogues")
        if not isinstance(dialogues, dict) or not dialogues:
            rejected.append(
                {
                    "variant": "multi",
                    "source_record_id": problem_id,
                    "reason": "empty_or_invalid_dialogues",
                }
            )
            continue

        for dialogue_id, turns in dialogues.items():
            locator = {
                "variant": "multi",
                "source_problem_id": problem_id,
                "source_record_id": dialogue_id,
                "split": split,
            }
            if not isinstance(turns, list) or not turns:
                rejected.append({"reason": "empty_or_invalid_dialogue", **locator})
                continue
            messages: list[dict[str, str]] = []
            user_types: list[str] = []
            bad_reason: str | None = None
            for turn_index, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    bad_reason = "invalid_turn_schema"
                    break
                teacher = normalize_text(turn.get("system"))
                student = normalize_text(turn.get("user"))
                if teacher:
                    messages.append({"role": "assistant", "content": teacher})
                elif "system" in turn:
                    empty_turns_dropped += 1
                if student:
                    messages.append({"role": "user", "content": student})
                    user_types.append(normalize_text(turn.get("user_type")))
                elif "user" in turn:
                    empty_turns_dropped += 1
                if not teacher and not student:
                    bad_reason = "turn_without_nonempty_message"
                    rejected.append(
                        {"reason": bad_reason, "source_turn_index": turn_index, **locator}
                    )
                    break
            if bad_reason:
                continue
            if not messages:
                rejected.append({"reason": "empty_normalized_dialogue", **locator})
                continue

            record = {
                "dataset": "SocraTeach",
                "variant": "multi_dialogue",
                "split": split,
                "source_record_id": dialogue_id,
                "source_problem_id": problem_id,
                "benchmark": benchmark,
                "messages": messages,
                "metadata": {
                    "question": normalize_text(problem["question"]),
                    "analysis": normalize_text(problem["analysis"]),
                    "answer": normalize_text(problem["answer"]),
                    "steps": [normalize_text(x) for x in problem["steps"]],
                    "student_user_types": user_types,
                },
                "provenance": {
                    "source_repository": "Ljyustc/SocraticLM",
                    "source_file": "github/SocraticLM/data/SocraTeach_multi.json",
                    "source_kind": "socratic_math_tutoring_dialogue",
                    "license": "CC BY-NC 4.0",
                    "commercial_use_warning": "Non-commercial license.",
                    "synthetic_model": None,
                    "synthetic_model_evidence": "Not specified in the local README/data files.",
                },
            }
            key = canonical_json({"split": split, "messages": messages})
            if key in multi_seen:
                rejected.append(
                    {
                        "reason": "exact_normalized_duplicate",
                        "duplicate_of": multi_seen[key],
                        **locator,
                    }
                )
                continue
            multi_seen[key] = dialogue_id
            multi_records.append(record)
            valid_dialogues.add(dialogue_id)
            multi_split_counts[split] += 1
            dialogue_turn_counts.append(len(messages))
            role_counts.update(x["role"] for x in messages)

    single_relation_counts: Counter[str] = Counter()
    for source_id, row in single_raw.items():
        match = SINGLE_ID_RE.fullmatch(source_id)
        if not match:
            rejected.append(
                {
                    "variant": "single",
                    "source_record_id": source_id,
                    "reason": "invalid_single_record_id",
                }
            )
            continue
        response_type, problem_id, benchmark, split, dialogue_index, turn_index, aug = (
            match.groups()
        )
        source_dialogue_id = f"{problem_id}_{dialogue_index}"
        locator = {
            "variant": "single",
            "source_record_id": source_id,
            "source_problem_id": problem_id,
            "source_dialogue_id": source_dialogue_id,
            "split": split,
        }
        if not isinstance(row, dict) or set(row) != {"prompt", "response", "history"}:
            rejected.append({"reason": "invalid_single_schema", **locator})
            continue
        history, bad_reason = normalize_pair_history(row.get("history"))
        if bad_reason:
            rejected.append({"reason": bad_reason, **locator})
            continue
        prompt = normalize_text(row.get("prompt"))
        response = normalize_text(row.get("response"))
        if not prompt or not response:
            rejected.append(
                {
                    "reason": "empty_prompt" if not prompt else "empty_response",
                    **locator,
                }
            )
            continue
        messages = history + [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        relation = (
            "linked_to_retained_multi_dialogue"
            if source_dialogue_id in valid_dialogues
            else "derived_from_socrateach_family_unlinked_to_retained_multi_id"
        )
        record = {
            "dataset": "SocraTeach",
            "variant": "single_turn_augmentation",
            "split": split,
            "source_record_id": source_id,
            "source_problem_id": problem_id,
            "source_dialogue_id": source_dialogue_id,
            "source_turn_index": int(turn_index),
            "augmentation_index": aug or None,
            "response_type": response_type,
            "benchmark": benchmark,
            "messages": messages,
            "provenance": {
                "source_repository": "Ljyustc/SocraticLM",
                "source_file": "github/SocraticLM/data/SocraTeach_single.json",
                "source_kind": "derived_single_turn_augmentation",
                "relationship_to_multi": relation,
                "license": "CC BY-NC 4.0",
                "commercial_use_warning": "Non-commercial license.",
                "synthetic_model": None,
                "synthetic_model_evidence": "Not specified in the local README/data files.",
            },
        }
        key = canonical_json(
            {
                "split": split,
                "response_type": response_type,
                "messages": messages,
            }
        )
        if key in single_seen:
            rejected.append(
                {
                    "reason": "exact_normalized_duplicate",
                    "duplicate_of": single_seen[key],
                    **locator,
                }
            )
            continue
        single_seen[key] = source_id
        single_records.append(record)
        single_split_counts[split] += 1
        single_relation_counts[relation] += 1

    write_jsonl(out_dir / "socrateach_multi.jsonl", multi_records)
    write_jsonl(out_dir / "socrateach_single.jsonl", single_records)
    cleaning.write_pool(
        "pedagogical_action",
        "SocraTeach",
        multi_records + single_records,
        removed={"rejected": [x for x in rejected if x["variant"] in {"multi", "single"}]},
    )
    write_jsonl(
        out_dir / "socrateach_rejected.jsonl",
        (x for x in rejected if x["variant"] in {"multi", "single"}),
    )
    report = {
        "dataset": "SocraTeach",
        "scope": "deterministic_structural_clean_no_utterance_rewrite_no_generation",
        "source_files": [source_file_entry(multi_path), source_file_entry(single_path)],
        "input": {
            "multi_problems": len(multi_raw),
            "multi_dialogues": sum(len(x["dialogues"]) for x in multi_raw.values()),
            "single_turn_augmentations": len(single_raw),
        },
        "kept": {
            "multi_dialogues": len(multi_records),
            "multi_split_counts": dict(sorted(multi_split_counts.items())),
            "single_turn_augmentations": len(single_records),
            "single_split_counts": dict(sorted(single_split_counts.items())),
            "single_multi_relationship_counts": dict(sorted(single_relation_counts.items())),
        },
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "empty_message_fields_dropped_before_row_validation": empty_turns_dropped,
        "normalization": [
            "CRLF/CR to LF and outer whitespace trim",
            "map source system to assistant and source user to user",
            "drop empty message fields; reject a source turn if no nonempty message remains",
            "retain all nonempty utterance wording unchanged",
        ],
        "dedup_policy": {
            "multi": "Exact equality of normalized split plus complete message sequence.",
            "single": "Exact equality of normalized split, response type, and complete history/prompt/response message sequence.",
            "cross_variant": "No cross-variant deletion: single is a derived augmentation view of multi, not an independent dialogue corpus.",
        },
        "counting_policy": (
            "Use multi_dialogues for unique full-dialogue corpus counts. Report single "
            "only as derived turn-level augmentation examples; never add multi and "
            "single counts as if they were independent source dialogues."
        ),
        "split_policy": "Split parsed from stable source IDs (train/test); no split invented.",
        "schema_checks": {
            "multi_roles_limited_to_user_assistant": set(role_counts)
            <= {
                "user",
                "assistant",
            },
            "multi_nonempty_messages": all(
                x["content"] for r in multi_records for x in r["messages"]
            ),
            "multi_turn_count_min": min(dialogue_turn_counts),
            "multi_turn_count_max": max(dialogue_turn_counts),
            "multi_role_counts": dict(sorted(role_counts.items())),
            "single_nonempty_messages": all(
                x["content"] for r in single_records for x in r["messages"]
            ),
        },
        "provenance_audit": {
            "audit_status": "documented",
            "license": "CC BY-NC 4.0",
            "commercial_use_warning": "Non-commercial license.",
            "source_repository": "Ljyustc/SocraticLM",
            "synthetic_model": None,
            "synthetic_model_status": "not specified in the local README/data files",
            "evidence_files": [
                "github/SocraticLM/README.md",
                "github/SocraticLM/LICENSE/DATA_LICENSE",
            ],
        },
    }
    write_json(out_dir / "socrateach_report.json", report)
    return report


def clean_convolearn(out_dir: Path) -> dict[str, Any]:
    paths = sorted((CONVOLEARN_DIR / "data").glob("*.parquet"))
    if not paths:
        raise ValueError("ConvoLearn: no parquet files found")
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    role_counts: Counter[str] = Counter()
    turn_counts: list[int] = []

    for source in paths:
        table = pq.read_table(source)
        expected = {
            "kb_subdim",
            "kb_dim",
            "effectiveness_consensus",
            "completeness_consensus",
            "cleaned_conversation",
            "earthscience_topic",
            "num_exchanges",
        }
        if set(table.column_names) != expected:
            raise ValueError(f"ConvoLearn schema mismatch: {table.column_names}")
        for row_number, row in enumerate(table.to_pylist(), 1):
            source_id = f"train:{row_number}"
            locator = {
                "source_file": str(source.relative_to(ACTION_ROOT)),
                "source_row_number": row_number,
                "source_record_id": source_id,
                "split": "train",
            }
            raw_conversation = normalize_text(row.get("cleaned_conversation"))
            if not raw_conversation:
                rejected.append({"reason": "empty_conversation", **locator})
                continue
            messages: list[dict[str, str]] = []
            bad_reason: str | None = None
            for line_number, line in enumerate(raw_conversation.split("\n"), 1):
                if not line.strip():
                    continue
                match = CONVO_LINE_RE.fullmatch(line)
                if not match:
                    bad_reason = "invalid_or_unknown_role_line"
                    rejected.append(
                        {
                            "reason": bad_reason,
                            "source_line_number": line_number,
                            **locator,
                        }
                    )
                    break
                source_role, content = match.groups()
                content = normalize_text(content)
                if not content:
                    bad_reason = "empty_turn_content"
                    rejected.append(
                        {
                            "reason": bad_reason,
                            "source_line_number": line_number,
                            **locator,
                        }
                    )
                    break
                messages.append(
                    {
                        "role": "user" if source_role == "Student" else "assistant",
                        "content": content,
                    }
                )
            if bad_reason:
                continue
            if not messages:
                rejected.append({"reason": "empty_normalized_conversation", **locator})
                continue
            record = {
                "dataset": "ConvoLearn",
                "split": "train",
                "source_record_id": source_id,
                "messages": messages,
                "metadata": {
                    "kb_subdim": normalize_text(row["kb_subdim"]),
                    "kb_dim": normalize_text(row["kb_dim"]),
                    "effectiveness_consensus": row["effectiveness_consensus"],
                    "completeness_consensus": row["completeness_consensus"],
                    "earthscience_topic": normalize_text(row["earthscience_topic"]),
                    "num_exchanges": row["num_exchanges"],
                },
                "provenance": {
                    "source_repository": "masharma/convolearn",
                    "source_file": str(source.relative_to(ACTION_ROOT)),
                    "source_kind": "real_teacher_and_simulated_student_dialogue",
                    "teacher_origin": "credentialed K-12 educators recruited through Prolific",
                    "student_origin": "simulated 7th-grade student",
                    "synthetic_model": "Gemini-1.5-Pro",
                    "annotation_models": ["GPT-4o", "Claude Haiku", "Claude Sonnet 4.5"],
                    "license": "MIT",
                },
            }
            key = canonical_json({"split": "train", "messages": messages})
            if key in seen:
                rejected.append(
                    {
                        "reason": "exact_normalized_duplicate",
                        "duplicate_of": seen[key],
                        **locator,
                    }
                )
                continue
            seen[key] = source_id
            records.append(record)
            turn_counts.append(len(messages))
            role_counts.update(x["role"] for x in messages)

    write_jsonl(out_dir / "convolearn.jsonl", records)
    write_jsonl(out_dir / "convolearn_rejected.jsonl", rejected)
    cleaning.write_pool(
        "pedagogical_action",
        "ConvoLearn",
        records,
        removed={"rejected": rejected},
    )
    report = {
        "dataset": "ConvoLearn",
        "scope": "deterministic_structural_clean_no_utterance_rewrite_no_generation",
        "source_files": [source_file_entry(x) for x in paths],
        "input_rows": len(records) + len(rejected),
        "kept_rows": len(records),
        "kept_split_counts": {"train": len(records)},
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "normalization": [
            "CRLF/CR to LF and outer whitespace trim",
            "parse one source role-prefixed line per turn",
            "map Student to user and Teacher to assistant",
            "retain all nonempty utterance wording unchanged",
        ],
        "dedup_policy": "Exact equality of normalized split plus complete message sequence.",
        "split_policy": "Official local parquet is train; train preserved and no new split invented.",
        "schema_checks": {
            "roles_limited_to_user_assistant": set(role_counts)
            <= {
                "user",
                "assistant",
            },
            "kept_nonempty_messages": all(x["content"] for r in records for x in r["messages"]),
            "turn_count_min": min(turn_counts),
            "turn_count_max": max(turn_counts),
            "role_counts": dict(sorted(role_counts.items())),
        },
        "provenance_audit": {
            "audit_status": "documented",
            "license": "MIT",
            "source_repository": "masharma/convolearn",
            "teacher_origin": "credentialed K-12 educators recruited through Prolific",
            "student_origin": "simulated 7th-grade student using Gemini-1.5-Pro",
            "annotation_models": ["GPT-4o", "Claude Haiku", "Claude Sonnet 4.5"],
            "evidence_files": ["huggingface/ConvoLearn/README.md"],
        },
    }
    write_json(out_dir / "convolearn_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    socrateach = clean_socrateach(args.output_dir)
    convolearn = clean_convolearn(args.output_dir)
    write_json(
        args.output_dir / "summary.json",
        {
            "scope": "fast structural clean and within-dataset exact dedup",
            "no_rewrite": True,
            "socrateach": socrateach["kept"],
            "convolearn": {
                "kept_rows": convolearn["kept_rows"],
                "kept_split_counts": convolearn["kept_split_counts"],
            },
        },
    )
    print(json.dumps(json.loads((args.output_dir / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
