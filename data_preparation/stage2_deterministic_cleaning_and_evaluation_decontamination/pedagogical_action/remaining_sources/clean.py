#!/usr/bin/env python3
"""Fast deterministic clean for remaining high-value pedagogical-action data.

Selected sources:
1. Eedi QATD-2k: real tutor/student dialogue, converted to tutor-next-turn SFT.
2. MathTutorBench MRBench train: chosen pedagogical response converted to SFT.

No utterance is generated or semantically rewritten. Operations are limited to
line-ending/outer-whitespace normalization, role mapping, adjacent same-role
message joining, structural validation, exact normalized deduplication, and
provenance/audit bookkeeping.
"""

from __future__ import annotations

import sys

import hashlib
import json
import re
import sys as _sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools

SOURCE_ROOT = pools.source_dir("pedagogical_action")
EEDI_ROOT = SOURCE_ROOT / "huggingface" / "Eedi-QATD-2k"
MTB_ROOT = SOURCE_ROOT / "huggingface" / "MathTutorBench-pedagogical-rewardmodel-data"
EEDI_OUT = pools.cleaned_dir("pedagogical_action", "Eedi")
MRBENCH_OUT = pools.cleaned_dir("pedagogical_action", "MRBench")


def norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, value: Any) -> str:
    return prefix + "_" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()[:20]


def file_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.relative_to(SOURCE_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


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


def exact_alias_groups(paths: list[Path]) -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[file_info(path)["sha256"]].append(path)
    return [
        {
            "sha256": digest,
            "files": [str(p.relative_to(SOURCE_ROOT)) for p in sorted(group)],
            "relation": "byte_identical_alias" if len(group) > 1 else "unique_file",
        }
        for digest, group in sorted(groups.items())
    ]


def question_metadata() -> (
    tuple[dict[tuple[int, int], dict[str, Any]], dict[int, list[dict[str, Any]]]]
):
    qdf = pd.read_csv(EEDI_ROOT / "dq-question-metadata.csv")
    sdf = pd.read_csv(EEDI_ROOT / "dialogue-subjects.csv")
    questions: dict[tuple[int, int], dict[str, Any]] = {}
    for (intervention_id, qid), frame in qdf.groupby(
        ["InterventionId", "QuestionId_DQ"], sort=False
    ):
        ordered = frame.sort_values(["Sequence", "MetaDataId"])
        questions[(int(intervention_id), int(qid))] = {
            "question_parts": [
                {"label": norm(row.Label), "text": norm(row.Text)}
                for row in ordered.itertuples()
                if norm(row.Text)
            ]
        }
    subjects: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sdf.sort_values(["InterventionId", "SubjectLevel", "SubjectId"]).itertuples():
        subjects[int(row.InterventionId)].append(
            {
                "name": norm(row.SubjectName),
                "level": int(row.SubjectLevel),
                "type": norm(row.SubjectType),
            }
        )
    return questions, subjects


def clean_eedi() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata, subjects = question_metadata()
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    talk_moves: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    input_rows = 0
    input_dialogues = 0
    empty_messages = 0
    adjacent_joins = 0

    source_paths = []
    for split in ("train", "test"):
        path = EEDI_ROOT / "anchored-dialogues" / f"{split}-00000-of-00001.parquet"
        source_paths.append(path)
        df = pq.read_table(path).to_pandas()
        input_rows += len(df)
        input_dialogues += df["InterventionId"].nunique()
        for intervention_id, frame in df.groupby("InterventionId", sort=False):
            frame = frame.sort_values("MessageSequence")
            qids = {int(x) for x in frame["QuestionId_DQ"].dropna().tolist()}
            if len(qids) != 1:
                rejected.append(
                    {
                        "dataset": "Eedi-QATD-2k",
                        "split": split,
                        "source_record_id": str(intervention_id),
                        "reason": "non_unique_question_id",
                        "question_ids": sorted(qids),
                    }
                )
                continue
            qid = next(iter(qids))
            merged: list[dict[str, Any]] = []
            for row in frame.itertuples():
                text = norm(row.MessageString)
                if not text:
                    empty_messages += 1
                    continue
                role = "assistant" if int(row.IsTutor) == 1 else "user"
                talk_move = norm(row.TalkMovePrediction) if role == "assistant" else ""
                source_seq = int(row.MessageSequence)
                if merged and merged[-1]["role"] == role:
                    merged[-1]["content"] += "\n\n" + text
                    merged[-1]["source_sequences"].append(source_seq)
                    if talk_move:
                        merged[-1]["talk_moves"].append(talk_move)
                    adjacent_joins += 1
                else:
                    merged.append(
                        {
                            "role": role,
                            "content": text,
                            "source_sequences": [source_seq],
                            "talk_moves": [talk_move] if talk_move else [],
                        }
                    )

            emitted = 0
            for index, message in enumerate(merged):
                if message["role"] != "assistant" or index == 0:
                    continue
                if merged[index - 1]["role"] != "user":
                    continue
                messages = [
                    {"role": m["role"], "content": m["content"]} for m in merged[: index + 1]
                ]
                key = canonical({"split": split, "messages": messages})
                record_id = stable_id("eedi", {"split": split, "messages": messages})
                if key in seen:
                    rejected.append(
                        {
                            "dataset": "Eedi-QATD-2k",
                            "split": split,
                            "source_record_id": f"{intervention_id}:{message['source_sequences'][-1]}",
                            "reason": "exact_normalized_sft_duplicate",
                            "duplicate_of": seen[key],
                        }
                    )
                    continue
                seen[key] = record_id
                moves = [x for x in message["talk_moves"] if x and x != "<None>"]
                talk_moves.update(moves or ["<None>"])
                role_counts.update(x["role"] for x in messages)
                records.append(
                    {
                        "dataset": "Eedi-QATD-2k",
                        "record_id": record_id,
                        "split": split,
                        "messages": messages,
                        "metadata": {
                            "intervention_id": int(intervention_id),
                            "question_id": qid,
                            "target_source_sequences": message["source_sequences"],
                            "target_talk_moves": moves,
                            "question": metadata.get((int(intervention_id), qid), {}).get(
                                "question_parts", []
                            ),
                            "subjects": subjects.get(int(intervention_id), []),
                        },
                        "provenance": {
                            "source_repository": "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
                            "source_file": str(path.relative_to(SOURCE_ROOT)),
                            "source_locator": f"InterventionId={intervention_id},MessageSequence={message['source_sequences'][-1]}",
                            "source_kind": "real_math_tutoring_dialogue",
                            "license": "CC BY-NC-SA 4.0",
                            "commercial_use_warning": "Non-commercial research use; contact Eedi for commercial use.",
                            "talk_move_annotation": "GPT-applied source labels",
                        },
                    }
                )
                emitted += 1
                split_counts[split] += 1
            if emitted == 0:
                rejected.append(
                    {
                        "dataset": "Eedi-QATD-2k",
                        "split": split,
                        "source_record_id": str(intervention_id),
                        "reason": "no_user_to_tutor_supervision_after_normalization",
                    }
                )

    test_message_keys = {canonical(r["messages"]) for r in records if r["split"] == "test"}
    no_leak_records: list[dict[str, Any]] = []
    for record in records:
        message_key = canonical(record["messages"])
        if record["split"] == "train" and message_key in test_message_keys:
            rejected.append(
                {
                    "dataset": "Eedi-QATD-2k",
                    "split": "train",
                    "source_record_id": record["provenance"]["source_locator"],
                    "reason": "cross_split_exact_duplicate_train_removed",
                    "test_split_preserved": True,
                }
            )
            continue
        no_leak_records.append(record)
    records = no_leak_records
    split_counts = Counter(x["split"] for x in records)
    talk_moves = Counter()
    role_counts = Counter()
    for record in records:
        talk_moves.update(record["metadata"]["target_talk_moves"] or ["<None>"])
        role_counts.update(x["role"] for x in record["messages"])

    report = {
        "dataset": "Eedi-QATD-2k",
        "scope": "deterministic tutor-next-turn SFT extraction; no utterance rewrite",
        "input_rows": input_rows,
        "input_dialogues": input_dialogues,
        "kept_records": len(records),
        "kept_split_counts": dict(sorted(split_counts.items())),
        "rejected_audit_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "normalization": [
            "CRLF/CR to LF and outer whitespace trim",
            "IsTutor=0 -> user; IsTutor=1 -> assistant",
            "adjacent same-role source messages joined with a blank line",
            "emit every normalized assistant turn immediately following a user turn",
        ],
        "empty_source_messages_dropped": empty_messages,
        "adjacent_same_role_joins": adjacent_joins,
        "exact_dedup_key": (
            "complete normalized SFT message sequence within split; exact train/test "
            "overlap removed from train while preserving official test"
        ),
        "target_talk_move_counts": dict(sorted(talk_moves.items())),
        "schema_checks": {
            "roles": dict(sorted(role_counts.items())),
            "all_records_end_assistant": all(
                r["messages"][-1]["role"] == "assistant" for r in records
            ),
            "all_targets_follow_user": all(r["messages"][-2]["role"] == "user" for r in records),
            "all_messages_nonempty": all(m["content"] for r in records for m in r["messages"]),
        },
        "split_policy": "Official train/test preserved; test is not relabeled as train.",
        "source_files": [
            file_info(x)
            for x in source_paths
            + [
                EEDI_ROOT / "dq-question-metadata.csv",
                EEDI_ROOT / "dialogue-subjects.csv",
                EEDI_ROOT / "README.md",
            ]
        ],
    }
    return records, rejected, report


def clean_mrbench() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data_dir = MTB_ROOT / "data"
    canonical_path = data_dir / "mrbench_train-00000-of-00001.parquet"
    rows = pq.read_table(canonical_path).to_pylist()
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    role_counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        history = row.get("dialog_history")
        if not isinstance(history, list) or not history:
            rejected.append(
                {
                    "dataset": "MathTutorBench-MRBench",
                    "split": "train",
                    "source_record_id": str(row_index),
                    "reason": "empty_or_invalid_history",
                }
            )
            continue
        messages: list[dict[str, str]] = []
        bad = False
        for item in history:
            if not isinstance(item, dict):
                bad = True
                break
            source_role = norm(item.get("user")).lower()
            role = (
                "assistant"
                if source_role in {"teacher", "tutor"}
                else "user" if source_role == "student" else ""
            )
            text = norm(item.get("text"))
            if not role or not text:
                bad = True
                break
            messages.append({"role": role, "content": text})
        target = norm(row.get("teacher_response_positive"))
        if bad or not target:
            rejected.append(
                {
                    "dataset": "MathTutorBench-MRBench",
                    "split": "train",
                    "source_record_id": str(row_index),
                    "reason": "invalid_role_or_empty_message",
                }
            )
            continue
        messages.append({"role": "assistant", "content": target})
        key = canonical({"split": "train", "messages": messages})
        record_id = stable_id("mrbench", {"split": "train", "messages": messages})
        if key in seen:
            rejected.append(
                {
                    "dataset": "MathTutorBench-MRBench",
                    "split": "train",
                    "source_record_id": str(row_index),
                    "reason": "exact_normalized_sft_duplicate",
                    "duplicate_of": seen[key],
                }
            )
            continue
        seen[key] = record_id
        role_counts.update(x["role"] for x in messages)
        records.append(
            {
                "dataset": "MathTutorBench-MRBench",
                "record_id": record_id,
                "split": "train",
                "messages": messages,
                "metadata": {
                    "problem": norm(row.get("problem")),
                    "topic": norm(row.get("topic")),
                    "reference_solution": norm(row.get("reference_solution")),
                    "rejected_teacher_response": norm(row.get("teacher_response_negative")),
                },
                "provenance": {
                    "source_repository": "eth-nlped/math-tutor-bench",
                    "source_dataset": "MRBench preference augmentation",
                    "source_file": str(canonical_path.relative_to(SOURCE_ROOT)),
                    "source_locator": f"row={row_index}",
                    "source_kind": "pedagogical_preference_pair",
                    "license": "CC BY 4.0",
                    "selection": "teacher_response_positive used as SFT target; negative retained as metadata only",
                },
            }
        )

    all_mtb = sorted(data_dir.glob("*.parquet"))
    report = {
        "dataset": "MathTutorBench-MRBench",
        "scope": "MRBench train only; chosen response SFT extraction; no utterance rewrite",
        "input_rows": len(rows),
        "kept_records": len(records),
        "rejected_audit_rows": len(rejected),
        "rejection_reasons": dict(sorted(Counter(x["reason"] for x in rejected).items())),
        "normalization": [
            "CRLF/CR to LF and outer whitespace trim",
            "Student -> user; Teacher/Tutor -> assistant",
            "teacher_response_positive appended as assistant SFT target",
            "teacher_response_negative retained as non-training metadata",
        ],
        "exact_dedup_key": "official split plus complete normalized SFT message sequence",
        "schema_checks": {
            "roles": dict(sorted(role_counts.items())),
            "all_records_end_assistant": all(
                r["messages"][-1]["role"] == "assistant" for r in records
            ),
            "all_messages_nonempty": all(m["content"] for r in records for m in r["messages"]),
        },
        "split_policy": (
            "Use mrbench_train as the canonical training file. Do not ingest the "
            "official test split. MathDial and GSM8K configurations are excluded."
        ),
        "alias_audit": exact_alias_groups(all_mtb),
        "canonical_source_file": file_info(canonical_path),
        "readme_file": file_info(MTB_ROOT / "README.md"),
    }
    return records, rejected, report


def clean() -> dict[str, Any]:
    eedi_records, eedi_rejected, eedi_report = clean_eedi()
    mr_records, mr_rejected, mr_report = clean_mrbench()
    write_jsonl(EEDI_OUT / "kept.jsonl", eedi_records)
    write_jsonl(EEDI_OUT / "rejected.jsonl", eedi_rejected)
    write_json(EEDI_OUT / "report.json", eedi_report)
    write_jsonl(MRBENCH_OUT / "kept.jsonl", mr_records)
    write_jsonl(MRBENCH_OUT / "rejected.jsonl", mr_rejected)
    write_json(MRBENCH_OUT / "report.json", mr_report)
    return {
        "Eedi-QATD-2k": {
            "records": len(eedi_records),
            "split_counts": dict(sorted(Counter(x["split"] for x in eedi_records).items())),
        },
        "MathTutorBench-MRBench": {
            "records": len(mr_records),
            "split_counts": {"train": len(mr_records)},
        },
    }


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(path: Path, expected_dataset: str) -> dict:
    rows = load_jsonl(path)
    ids = [x["record_id"] for x in rows]
    keys = [
        json.dumps(
            {"split": x["split"], "messages": x["messages"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for x in rows
    ]
    assert len(ids) == len(set(ids))
    assert len(keys) == len(set(keys))
    assert all(x["dataset"] == expected_dataset for x in rows)
    assert all(x["messages"][-1]["role"] == "assistant" for x in rows)
    assert all(
        m["role"] in {"user", "assistant"} and m["content"].strip()
        for x in rows
        for m in x["messages"]
    )
    return {"rows": len(rows), "unique_ids": len(set(ids)), "unique_examples": len(set(keys))}


def validate() -> dict[str, Any]:
    eedi_rows = load_jsonl(EEDI_OUT / "kept.jsonl")
    result = {
        "eedi": check(EEDI_OUT / "kept.jsonl", "Eedi-QATD-2k"),
        "mrbench": check(MRBENCH_OUT / "kept.jsonl", "MathTutorBench-MRBench"),
    }
    by_messages: dict[str, set[str]] = {}
    for row in eedi_rows:
        key = json.dumps(
            row["messages"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        by_messages.setdefault(key, set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_messages.values())
    return result


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = clean()
    result = validate()
    print(json.dumps({"cleaned": summary, "validated": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
