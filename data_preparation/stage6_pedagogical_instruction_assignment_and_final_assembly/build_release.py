#!/usr/bin/env python3
"""Assemble the post-scoring/post-selection release for LLaMA-Factory."""

import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import sys as _sys

import cairosvg

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common.media import media_specs, parquet_column

FINAL = (
    Path("data_preparation/stage5_token_budgeted_diversity_selection")
    / "results/bucketed_kcenter_final"
)
OUT = (
    Path("data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/releases/assembled")
)
MEDIA = OUT / "media"

CATEGORIES = (
    "subject_competence",
    "diagnostic_reasoning",
    "curriculum_grounding",
    "pedagogical_action",
)
ARCHIVE_HANDLES = {}
TAR_MEMBERS = {}


def read_jsonl(path):
    with path.open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def options_text(options):
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{key}. {value}" for key, value in options.items())
    if isinstance(options, list):
        return "\n".join(f"{chr(65 + index)}. {value}" for index, value in enumerate(options))
    return text(options)


def clean_messages(messages):
    result = []
    for message in messages or []:
        role = str(message.get("role") or message.get("user") or "").lower()
        role = {
            "teacher": "assistant",
            "tutor": "assistant",
            "student": "user",
            "human": "user",
            "gpt": "assistant",
        }.get(role, role)
        content = text(message.get("content") or message.get("text")).strip()
        if role in {"system", "user", "assistant"} and content:
            result.append({"role": role, "content": content})
    if not result:
        return []
    systems = [item for item in result if item["role"] == "system"]
    dialogue = [item for item in result if item["role"] != "system"]
    if not dialogue or dialogue[-1]["role"] != "assistant":
        return []
    if dialogue[0]["role"] != "user" or any(
        dialogue[index]["role"] == dialogue[index - 1]["role"] for index in range(1, len(dialogue))
    ):
        target = dialogue[-1]["content"]
        context = "\n".join(
            f"[{'Tutor' if item['role'] == 'assistant' else 'Student'}] {item['content']}"
            for item in dialogue[:-1]
        )
        dialogue = [
            {"role": "user", "content": context or "Continue the tutoring interaction."},
            {"role": "assistant", "content": target},
        ]
    return systems[:1] + dialogue


def render_subject(row, payload):
    if row["kind"] == "essay":
        user = (
            f"Task:\n{text(payload.get('prompt'))}\n\nStudent response:\n"
            f"{text(payload.get('essay'))}"
        )
        if payload.get("scores"):
            user += f"\n\nExisting rubric scores:\n{text(payload['scores'])}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": text(payload.get("feedback"))},
        ]
    user_parts = []
    if payload.get("passage"):
        user_parts.append(f"Passage:\n{text(payload['passage'])}")
    user_parts.append(f"Problem:\n{text(payload.get('problem'))}")
    choices = options_text(payload.get("options"))
    if choices:
        user_parts.append(f"Options:\n{choices}")
    target = payload.get("cot") or payload.get("answer_text") or payload.get("answer")
    return [
        {"role": "user", "content": "\n\n".join(user_parts)},
        {"role": "assistant", "content": text(target)},
    ]


def render_longtutor(payload):
    if payload.get("messages"):
        return clean_messages(payload["messages"])
    annotation = payload.get("annotation") or {}
    target = {
        key: annotation.get(key)
        for key in ("diagnosis", "strategy", "reason", "memory", "content")
        if annotation.get(key) is not None
    }
    user = {
        "history": payload.get("history"),
        "current_question": payload.get("current_question"),
        "current_interaction": payload.get("current_interaction"),
    }
    return [
        {
            "role": "user",
            "content": "Analyze this learner history and produce the next longitudinal tutoring action.\n"
            + text(user),
        },
        {"role": "assistant", "content": text(target)},
    ]


def render_diagnostic(row, payload):
    source = row["source"]
    if row["kind"] == "longitudinal_tutoring":
        return render_longtutor(payload)
    if source == "MathDial":
        user = {
            "problem": payload.get("question"),
            "student_profile": payload.get("student_profile"),
            "student_incorrect_solution": payload.get("student_incorrect_solution"),
            "tutoring_dialogue": payload.get("conversation"),
            "teacher_described_confusion": payload.get("teacher_described_confusion"),
        }
        return [
            {
                "role": "user",
                "content": "Diagnose the remaining issue and provide the correct resolution.\n"
                + text(user),
            },
            {"role": "assistant", "content": text(payload.get("ground_truth"))},
        ]
    if source == "Bridge":
        user = {
            "lesson_topic": payload.get("lesson_topic"),
            "student_error": payload.get("student_error"),
            "conversation_history": payload.get("conversation_history"),
            "original_tutor_response": payload.get("original_tutor_response"),
            "strategy": payload.get("strategy"),
        }
        return [
            {
                "role": "user",
                "content": "Revise the tutor response using the learner context.\n" + text(user),
            },
            {
                "role": "assistant",
                "content": text(payload.get("expert_revised_response")),
            },
        ]
    if source == "ScratchMath":
        user = {
            "question": payload.get("question"),
            "student_answer": payload.get("student_answer"),
            "student_scratchwork": "<image>",
        }
        target = {
            "error_category": payload.get("error_category"),
            "error_explanation": payload.get("error_explanation"),
            "correct_solution": payload.get("solution"),
        }
        return [
            {"role": "user", "content": text(user)},
            {"role": "assistant", "content": text(target)},
        ]
    if source == "StepVerify":
        user = {
            "problem": payload.get("problem"),
            "dialog_history": payload.get("dialog_history"),
            "student_incorrect_solution": payload.get("student_incorrect_solution"),
        }
        target = {
            "error_category": payload.get("error_category"),
            "error_description": payload.get("error_description"),
            "correct_response": payload.get("student_correct_response"),
        }
        return [
            {"role": "user", "content": text(user)},
            {"role": "assistant", "content": text(target)},
        ]
    raise ValueError(f"unsupported diagnostic source: {source}")


def render_curriculum(payload):
    direct = clean_messages(payload.get("messages"))
    if direct:
        return direct
    return [
        {"role": "user", "content": text(payload.get("question"))},
        {"role": "assistant", "content": text(payload.get("answer"))},
    ]


def render_record(record):
    direct = clean_messages(record.get("messages"))
    if direct:
        return direct
    if record.get("hint"):
        user = "\n\n".join(
            part
            for part in (
                f"Passage:\n{text(record.get('passage'))}" if record.get("passage") else "",
                f"Question:\n{text(record.get('question'))}",
                "Give a useful hint without simply revealing the answer.",
            )
            if part
        )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": text(record["hint"])},
        ]
    if record.get("socratic_questions"):
        return [
            {
                "role": "user",
                "content": f"Problem:\n{text(record.get('problem'))}\n\nGuide me with Socratic questions.",
            },
            {"role": "assistant", "content": text(record["socratic_questions"])},
        ]
    target_key = next(
        (
            key
            for key in ("chosen_feedback", "feedback", "solution", "response", "answer")
            if record.get(key)
        ),
        None,
    )
    if not target_key:
        raise ValueError(f"no pedagogical target in {sorted(record)}")
    excluded = {
        target_key,
        "rejected_feedback",
        "provenance",
        "dedup",
        "privacy_safety",
        "source_metadata",
    }
    user = {key: value for key, value in record.items() if key not in excluded}
    return [
        {
            "role": "user",
            "content": "Provide the appropriate educational response to this learner context.\n"
            + text(user),
        },
        {"role": "assistant", "content": text(record[target_key])},
    ]


def render_pedagogy(row, payload):
    direct = clean_messages(payload.get("messages"))
    if direct:
        return direct
    if row["kind"] == "mathtutor_scaffolding":
        user = (
            "You are an experienced math teacher. Respond to the student usefully and caringly.\n"
            f"Problem: {text(payload.get('problem'))}\n"
            f"Student attempt: {text(payload.get('student_attempt'))}\nTeacher:"
        )
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": text(payload.get("teacher_turn"))},
        ]
    if row["kind"] in {"tutorbench_dialogue", "oatutor_guidance"}:
        messages = clean_messages(
            list(payload.get("context_messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": text(payload.get("candidate_teacher_turn")),
                }
            ]
        )
        return messages
    return render_record(payload.get("record") or payload)


def materialize_destination(spec):
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if spec["type"] == "file":
        source = Path(spec["path"])
        if source.suffix.lower() != ".svg":
            return source
        return MEDIA / key[:2] / f"{key}.png"
    if spec["type"] == "archive":
        suffix = Path(spec["member"]).suffix or ".bin"
        if suffix.lower() == ".svg":
            suffix = ".png"
    else:
        item = parquet_column(Path(spec["path"]), spec["column"])[spec["row_index"]][spec["column"]]
        if spec.get("list_index") is not None:
            item = item[spec["list_index"]]
        suffix = Path(item.get("path") or "").suffix or ".png"
    return MEDIA / key[:2] / f"{key}{suffix}"


def materialize(spec):
    destination = materialize_destination(spec)
    if spec["type"] == "file" and destination == Path(spec["path"]):
        return str(destination)
    if destination.exists():
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec["type"] == "file":
        data = Path(spec["path"]).read_bytes()
    elif spec["type"] == "archive":
        archive = Path(spec["archive"])
        if archive.suffix.lower() == ".zip":
            key = str(archive)
            if key not in ARCHIVE_HANDLES:
                ARCHIVE_HANDLES[key] = zipfile.ZipFile(archive)
            data = ARCHIVE_HANDLES[key].read(spec["member"])
        else:
            key = str(archive)
            if key not in ARCHIVE_HANDLES:
                handle = tarfile.open(archive)
                ARCHIVE_HANDLES[key] = handle
                TAR_MEMBERS[key] = {member.name: member for member in handle.getmembers()}
            member = ARCHIVE_HANDLES[key].extractfile(TAR_MEMBERS[key][spec["member"]])
            if member is None:
                raise ValueError(f"cannot extract {spec}")
            data = member.read()
    elif spec["type"] == "parquet":
        item = parquet_column(Path(spec["path"]), spec["column"])[spec["row_index"]][spec["column"]]
        if spec.get("list_index") is not None:
            item = item[spec["list_index"]]
        data = item["bytes"]
    else:
        raise ValueError(spec)
    source_suffix = (
        Path(spec["path"]).suffix
        if spec["type"] == "file"
        else Path(spec.get("member") or "").suffix
    )
    if source_suffix.lower() == ".svg":
        data = cairosvg.svg2png(bytestring=data)
    destination.write_bytes(data)
    return str(destination)


def materialize_all(selected_rows):
    specs = {}
    for row in selected_rows:
        for spec in media_specs(row):
            specs[json.dumps(spec, sort_keys=True)] = spec
    tar_groups = {}
    other = []
    for spec in specs.values():
        if (
            spec["type"] == "archive"
            and Path(spec["archive"]).suffix.lower() != ".zip"
            and not materialize_destination(spec).exists()
        ):
            tar_groups.setdefault(spec["archive"], {})[spec["member"]] = spec
        else:
            other.append(spec)
    for archive_name, wanted in tar_groups.items():
        remaining = set(wanted)
        with tarfile.open(archive_name) as handle:
            for member in handle:
                if member.name not in remaining:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot extract {archive_name}::{member.name}")
                destination = materialize_destination(wanted[member.name])
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = source.read()
                if Path(member.name).suffix.lower() == ".svg":
                    data = cairosvg.svg2png(bytestring=data)
                destination.write_bytes(data)
                remaining.remove(member.name)
                if not remaining:
                    break
        if remaining:
            raise ValueError(f"{archive_name}: missing {len(remaining)} selected members")
    for spec in other:
        materialize(spec)
    print(f"materialized media specs: {len(specs)}", flush=True)


def attach_images(messages, images):
    if not images:
        return messages
    for message in messages:
        message["content"] = re.sub(r"###IMAGE\d+###", "<image>", message["content"])
    present = sum(message["content"].count("<image>") for message in messages)
    if present < len(images):
        first_user = next(message for message in messages if message["role"] == "user")
        first_user["content"] = "<image>" * (len(images) - present) + "\n" + first_user["content"]
    elif present > len(images):
        raise ValueError(f"image token/spec mismatch: tokens={present} images={len(images)}")
    return messages


def render(row):
    payload = row["payload"]
    if row["category"] == "subject_competence":
        messages = render_subject(row, payload)
    elif row["category"] == "diagnostic_reasoning":
        messages = render_diagnostic(row, payload)
    elif row["category"] == "curriculum_grounding":
        messages = render_curriculum(payload)
    else:
        messages = render_pedagogy(row, payload)
    messages = clean_messages(messages)
    if not messages:
        raise ValueError(f"invalid messages: {row['category']}::{row['source']}::{row['uid']}")
    images = [materialize(spec) for spec in media_specs(row)]
    messages = attach_images(messages, images)
    extra = {
        "category": row["category"],
        "dataset": row["source"],
        "uid": row["uid"],
        "kind": row["kind"],
        "kcenter": row["kcenter"],
    }
    return {
        "messages": messages,
        "images": images,
        "extra_info": json.dumps(extra, ensure_ascii=False),
    }


def validate(train_rows):
    seen = set()
    counts = Counter()
    for index, row in enumerate(train_rows):
        messages = row["messages"]
        assert messages and messages[-1]["role"] == "assistant", index
        assert messages[-1]["content"].strip(), index
        assert any(message["role"] == "user" for message in messages), index
        image_tokens = sum(message["content"].count("<image>") for message in messages)
        assert image_tokens == len(row.get("images") or []), (index, image_tokens, row["images"])
        assert all(Path(path).exists() for path in row.get("images") or []), index
        info = (
            json.loads(row["extra_info"])
            if isinstance(row["extra_info"], str)
            else row["extra_info"]
        )
        uid = (
            info.get("uid")
            or hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
        )
        key = (info.get("category"), info.get("dataset"), uid)
        assert key not in seen, key
        seen.add(key)
        counts[info.get("category")] += 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    selected_by_category = {
        category: list(read_jsonl(FINAL / f"{category}.jsonl")) for category in CATEGORIES
    }
    materialize_all([row for category in CATEGORIES for row in selected_by_category[category]])
    output_rows = []
    for category in CATEGORIES:
        for row in selected_by_category[category]:
            output_rows.append(render(row))
    counts = validate(output_rows)
    with (OUT / "train.jsonl").open("w") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_info = {
        "omniedu_assembled": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    (OUT / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n"
    )
    report = {
        "total": len(output_rows),
        "counts": dict(sorted(counts.items())),
        "kcenter_source": str(FINAL),
    }
    (OUT / "assembly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
