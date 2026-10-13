#!/usr/bin/env python3
"""Build the final release with deterministic task/style-specific system prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    Path("data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/releases/assembled")
)
DEFAULT_OUTPUT = (
    Path("data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly/releases/omniedu")
)
DEFAULT_TOKENIZER = Path("models/Qwen3.5-9B-Base")
EDUCATION_ROWS = 60_951
GENERAL_ROWS = 9_048
COMBINED_ROWS = 69_999
GENERAL_POOL = Path(
    "data_preparation/stage1_capability_taxonomy_and_source_collection/general_purpose/"
    "general_instruction_pool.jsonl"
)

PROMPTS = {
    "solve_answer_only": (
        "Solve the problem independently and accurately using all provided text and visual "
        "evidence. Return only the final answer in the exact format requested, without explanation."
    ),
    "solve_reasoned": (
        "Solve the problem independently and accurately using all provided text and visual "
        "evidence. Give a clear, self-contained derivation and finish with one unambiguous final "
        "answer in the requested format."
    ),
    "reading_comprehension": (
        "Answer the reading-comprehension question from the supplied passage and options. Base the "
        "answer only on the passage, reason carefully when needed, and follow the requested answer format."
    ),
    "writing_feedback": (
        "Evaluate the student's writing against the supplied task and rubric. Give specific, "
        "evidence-based feedback that identifies strengths and the most useful improvements without "
        "inventing unsupported details."
    ),
    "diagnose_and_correct": (
        "Diagnose the learner's actual error from the supplied work or dialogue. Preserve correct "
        "work, identify the first material issue, explain why it matters, and provide the most useful "
        "next correction or resolution."
    ),
    "knowledge_point": (
        "This is a curriculum and knowledge-point task. Identify or judge the relevant concepts, "
        "standards, prerequisites, or relations from the provided information, including images when "
        "present, and follow the requested output format exactly."
    ),
    "mathfish_strict": (
        "Judge whether the proposed curriculum alignment is valid under the requested MathFish "
        "verification criterion. Answer Yes or No and give one brief, concrete reason."
    ),
    "mathfish_multirelation": (
        "Analyze the curriculum relationship requested by the MathFish task. Distinguish direct "
        "alignment from prerequisites, neighboring skills, and later applications, and return only "
        "the requested relation format."
    ),
    "mathtutor_scaffolding": (
        "You are an experienced math teacher responding to a student in a useful and caring way. "
        "Give the smallest helpful next scaffold and keep the response to at most two sentences."
    ),
    "mathtutor_pedagogy_following": (
        "Be a friendly, supportive tutor. Guide the student through incremental steps with guiding "
        "and probing questions. Pose just one question in this turn, avoid overwhelming the student, "
        "and keep the response to at most two sentences."
    ),
    "tutorbench": (
        "You are a highly capable, supportive tutor. Respond directly to the student. Be accurate, "
        "pedagogically helpful, appropriately calibrated to the student, and avoid unnecessary verbosity."
    ),
    "active_probe": (
        "Facilitate active learning with one actionable question that asks the learner to calculate, "
        "compare, retrieve, test, explain, or inspect something relevant. Do not reveal the final answer."
    ),
    "answer_assessment": (
        "Assess the learner's answer accurately. State what is correct, identify the first important "
        "error or omission, and give focused feedback or the next correction without unnecessary re-solving."
    ),
    "direct_explanation": (
        "Give a clear, accurate explanation adapted to the learner's request. Cover the essential idea "
        "and reasoning directly, using a useful example or representation when appropriate."
    ),
    "multihint": (
        "Provide a short sequence of progressively stronger hints. Begin with the least revealing useful "
        "hint, keep each hint actionable, and do not disclose the final answer unless explicitly requested."
    ),
    "multiturn_socratic": (
        "Tutor through a coherent Socratic dialogue. Use the conversation history, ask focused questions "
        "that build on the learner's responses, and advance one conceptual step at a time."
    ),
    "oatutor_guidance": (
        "Guide the learner with a targeted next step based on the current problem and work. Prefer a concise "
        "hint or question that makes the learner reason instead of giving away the final answer."
    ),
    "socratic_question_chain": (
        "Use a concise chain of logically ordered Socratic questions that leads the learner toward the key "
        "idea. Each question should have a clear purpose and must not reveal the final answer."
    ),
    "longtutor_official": (
        """### ROLE
You are an expert AI Tutor. Analyze student data to answer queries, diagnose state and provide targeted teaching.

### CONFIGURATION
[Diagnosis & Strategy Schema]
- **Recall Failure**: Forgot previously mastered knowledge (> 4 days).
  -> Strategy: **Retrieval Practice** (Guide student to recall specific steps or concepts from past successful exercises.)
- **Conceptual Gap**: Fundamental misunderstanding of the concept (New or complex topic).
  -> Strategy: **Conceptual Explanation** (Explain the underlying concept using analogies/definitions before solving.)
- **Procedural Error**: Understands concept but fails in calculation/steps (Complex problem or careless mistake).
  -> Strategy: **Stepwise Scaffolding** (Break down the problem into smaller steps and correct the specific wrong step.)
- **Transfer Deficit**: Fails to apply knowledge in a novel context or variation.
  -> Strategy: **Analogical Transfer** (Compare the current problem with a similar past solved problem to show connections.)

### TASKS
1. **Memory**: Answer specific queries based on Interaction History. If information is insufficient or premises are incorrect, respond "Unknown".
2. **Diagnosis**: Infer the student's state. Select ONE diagnosis from the Schema.
3. **Teaching**: Generate a teaching response executing the chosen strategy action. Optimize the response by evaluating it against four key dimensions: History Utilization, Strategy Alignment, Coherence, and Appropriateness.

### OUTPUT FORMAT
Return strictly a JSON object:
{
  "memory": [{"id": "Q1", "answer": "... (in Simple Chinese)"}, {"id": "Q2", "answer": "..."}, {"id": "Q3", "answer": "..."}]
  "diagnosis": "Recall Failure / Conceptual Gap / Procedural Error / Transfer Deficit (choose one, in English)",
  "reason": "Justification citing specific history or features (in Simple Chinese).",
  "strategy": "Analogical Transfer / Stepwise Scaffolding / Conceptual Explanation / Retrieval Practice (mapped to diagnosis, in English)",
  "content": "The teaching intervention content (in Simple Chinese)."
}
Make outputs concise."""
    ),
}

#: ?? Table ???? ? 19 ??????????? stage5 ??????
PEDAGOGY_BUCKETS = {
    "scaffolding/minimal_scaffold": "mathtutor_pedagogy_following",
    "scaffolding/hint_framed_as_a_question": "oatutor_guidance",
    "scaffolding/progressive_hints_no_question": "multihint",
    "questioning/one_question_per_turn": "multiturn_socratic",
    "questioning/ordered_question_chain": "socratic_question_chain",
    "multi_turn_dialogue": "tutorbench",
    "assessment/feedback_on_a_solution": "answer_assessment",
    "assessment/feedback_on_writing": "writing_feedback",
    "direct_explanation": "direct_explanation",
    "dialogue/long_conversation_history": "longtutor_official",
}


def read_rows(path: Path):
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank line")
            yield json.loads(line)


def extra_info(row: dict) -> dict:
    value = row.get("extra_info") or {}
    return json.loads(value) if isinstance(value, str) else dict(value)


def without_system(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("role") != "system"]


def target_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def mathtutor_prompt_id(messages: list[dict]) -> str:
    target = target_text(messages).strip()
    # The official pedagogy-following protocol requires exactly one question.
    # Only targets already exhibiting that behavior receive that instruction.
    question_count = target.count("?") + target.count("？")
    sentence_count = len(
        [piece for piece in re.split(r"(?<=[.!?。！？])\s+", target) if piece.strip()]
    )
    if question_count == 1 and sentence_count <= 2:
        return "mathtutor_pedagogy_following"
    return "mathtutor_scaffolding"


def choose_prompt(info: dict, messages: list[dict]) -> str:
    category = info.get("category")
    kcenter = info.get("kcenter") or {}
    bucket = kcenter.get("bucket", "")

    if category == "subject_competence":
        if "/answer_only" in bucket:
            return "solve_answer_only"
        if bucket.startswith("reading/"):
            return "reading_comprehension"
        if bucket.startswith("writing_feedback/"):
            return "writing_feedback"
        return "solve_reasoned"

    if category == "diagnostic_reasoning":
        if bucket.startswith("dialogue/long_conversation_history/"):
            return "longtutor_official"
        return "diagnose_and_correct"

    if category == "curriculum_grounding":
        if bucket == "exercise_standard_mapping/strict":
            return "mathfish_strict"
        if bucket == "exercise_standard_mapping/multirelation":
            return "mathfish_multirelation"
        return "knowledge_point"

    if category == "pedagogical_action":
        if bucket == "scaffolding/minimal_scaffold":
            return mathtutor_prompt_id(messages)
        if bucket == "scaffolding/hint_framed_as_a_question" and str(
            info.get("dataset") or info.get("source") or ""
        ) == "active-learning-hint":
            return "active_probe"
        if bucket in PEDAGOGY_BUCKETS:
            return PEDAGOGY_BUCKETS[bucket]
        raise KeyError(f"unmapped pedagogical bucket: {bucket}")

    if category == "general_instruct":
        return "native_or_general"
    raise KeyError(f"unmapped category: {category}")


def message_fingerprint(messages: list[dict]) -> str:
    payload = json.dumps(without_system(messages), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build_row(row: dict, registry: dict[str, str]) -> tuple[dict, str]:
    messages = row.get("messages") or []
    dialogue = without_system(messages)
    if not dialogue or dialogue[-1].get("role") != "assistant":
        raise ValueError("invalid dialogue")
    info = extra_info(row)
    prompt_id = choose_prompt(info, messages)
    if prompt_id == "native_or_general":
        existing = next(
            (
                str(message.get("content") or "").strip()
                for message in messages
                if message.get("role") == "system" and str(message.get("content") or "").strip()
            ),
            None,
        )
        # General-purpose examples keep their own system prompt; this stage only
        # assigns the task templates above to education-specific rows.
        if not existing:
            raise ValueError("general-purpose row carries no system prompt to retain")
        system = existing
        prompt_id = "general_native"
    else:
        system = registry[prompt_id]
    info["system_prompt_id"] = prompt_id
    result = dict(row)
    result["messages"] = [{"role": "system", "content": system}] + dialogue
    result["extra_info"] = json.dumps(info, ensure_ascii=False)
    return result, prompt_id


def token_report(rows: list[dict], tokenizer_path: Path, cutoff: int) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    lengths = []
    over = []
    batch_texts = []
    batch_indices = []

    def consume() -> None:
        if not batch_texts:
            return
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=False,
            truncation=False,
            return_length=True,
        )
        for index, length in zip(batch_indices, encoded["length"]):
            lengths.append(int(length))
            if length > cutoff:
                over.append({"index": index, "tokens": int(length)})
        batch_texts.clear()
        batch_indices.clear()

    for index, row in enumerate(rows):
        batch_texts.append(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
        )
        batch_indices.append(index)
        if len(batch_texts) == 64:
            consume()
    consume()
    ordered = sorted(lengths)
    return {
        "tokenizer": str(tokenizer_path),
        "cutoff": cutoff,
        "max_tokens": max(lengths),
        "p99_tokens": ordered[int(0.99 * (len(ordered) - 1))],
        "over_cutoff_count": len(over),
        "over_cutoff_examples": over[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--cutoff",
        type=int,
        default=32768,
        help="training max sequence length used for the over-length report",
    )
    parser.add_argument("--skip-token-report", action="store_true")
    args = parser.parse_args()

    source_path = args.input_dir / "train.jsonl"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = dict(PROMPTS)

    rows = []
    prompt_counts = Counter()
    source_fingerprints = []
    for source_row in read_rows(source_path):
        original_messages = source_row.get("messages") or []
        source_fingerprints.append(message_fingerprint(original_messages))
        row, prompt_id = build_row(source_row, registry)
        if message_fingerprint(row["messages"]) != source_fingerprints[-1]:
            raise AssertionError("non-system messages changed")
        if row["messages"][0]["role"] != "system":
            raise AssertionError("missing system prompt")
        rows.append(row)
        prompt_counts[prompt_id] += 1

    general = []
    for row in read_rows(GENERAL_POOL):
        info = extra_info(row)
        if info.get("category") == "general_instruct":
            general.append(row)
    if len(general) != GENERAL_ROWS:
        raise ValueError(f"expected {GENERAL_ROWS} general rows, got {len(general)}")

    final_rows = rows + general
    temporary = args.output_dir / "train.jsonl.partial"
    with temporary.open("w", encoding="utf-8") as output:
        for row in final_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output_dir / "train.jsonl")

    dataset_info = {
        "omniedu": {
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
    (args.output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "system_prompt_registry.json").write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "input": str(args.input_dir),
        "output": str(args.output_dir),
        "education_specific_rows": len(rows),
        "general_rows": len(general),
        "expected_education_specific_rows": EDUCATION_ROWS,
        "expected_general_rows": GENERAL_ROWS,
        "total": len(final_rows),
        "prompt_counts": dict(sorted(prompt_counts.items())),
        "all_rows_have_system": all(
            row["messages"] and row["messages"][0]["role"] == "system" for row in rows
        ),
        "general_pool": str(GENERAL_POOL),
        "non_system_content_unchanged": True,
        "images_unchanged": True,
    }
    if not args.skip_token_report:
        report["sequence_tokens"] = token_report(rows, args.tokenizer, args.cutoff)
    (args.output_dir / "assembly_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
