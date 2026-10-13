"""Fine-grained task buckets shared by stage 4 filtering and stage 5 selection.

Bucket names are also the token-budget keys of ``stage5_.../select_by_token_budget.py``, so changing
this mapping changes the final corpus. ``target_text`` extracts the text that is actually supervised,
which is used to measure supervised response tokens.
"""

from __future__ import annotations

import json

#: math sources (decide math/* vs science/*)
MATH_SOURCES = {
    "GSM8K",
    "MATH",
    "MathQA",
    "OpenR1",
    "SciInstruct (CN math)",
    "CMM-Math (text cot)",
    "CMM-Math (text answer)",
    "CMM-Math (image)",
    "Geometry3K",
}

#: image-bearing sources (decide text vs multimodal)
MM_SOURCES = {
    "CMM-Math (image)",
    "Geometry3K",
    "ScienceQA (image)",
    "TQA (diagram)",
    "AI2D",
    "ScratchMath",
    "CIMA",
}


def assistant_text(messages) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in (messages or [])
        if message.get("role") == "assistant"
    ).strip()


def stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def target_text(row: dict) -> str:
    """Extract the supervised output; multi-field tasks are merged into one text."""
    payload = row.get("payload") or {}
    direct = assistant_text(payload.get("messages"))
    if direct:
        return direct

    kind = row.get("kind")
    source = row.get("source")
    if row.get("category") == "subject_competence":
        return stringify(
            payload.get("cot")
            or payload.get("answer")
            or payload.get("answer_text")
            or payload.get("feedback")
        )
    if kind == "longitudinal_tutoring":
        return stringify(payload.get("annotation") or payload.get("content"))
    if source == "ScratchMath":
        return stringify(
            {
                "error_explanation": payload.get("error_explanation"),
                "solution": payload.get("solution"),
            }
        )
    if source == "StepVerify":
        return stringify(
            payload.get("student_correct_response")
            or payload.get("error_description")
            or payload.get("incorrect_step")
        )
    if source == "MathDial":
        return stringify(payload.get("conversation"))
    if source == "Bridge":
        return stringify(payload.get("expert_revised_response"))
    if row.get("category") == "curriculum_grounding":
        return stringify(payload.get("answer"))
    if kind in {"tutorbench_dialogue", "oatutor_guidance"}:
        return stringify(payload.get("candidate_teacher_turn"))
    if kind == "mathtutor_scaffolding":
        return stringify(payload.get("teacher_turn"))

    record = payload.get("record") or payload
    nested = assistant_text(record.get("messages")) if isinstance(record, dict) else ""
    if nested:
        return nested
    for key in (
        "candidate_teacher_turn",
        "teacher_turn",
        "chosen_feedback",
        "feedback",
        "hint",
        "socratic_questions",
        "solution",
        "response",
        "answer",
    ):
        if isinstance(record, dict) and record.get(key):
            return stringify(record[key])
    return stringify(record)


def bucket_id(category: str, source: str, kind: str) -> str:
    """????? Table~\ref{tab:bucket-budgets} ?????????? 32 ????"""
    if category == "subject_competence":
        if kind == "reading":
            return "reading/reading-race" if source == "RACE" else "reading/reading-c3"
        if kind == "essay":
            return "writing_feedback/cn" if source == "CSEE" else "writing_feedback/en"
        supervision = "answer_only" if kind == "answer_only" or "answer" in source.lower() else "cot"
        modality = "multimodal" if source in MM_SOURCES else "text"
        if source in MATH_SOURCES:
            return f"math/{modality}/{supervision}"
        return f"science/{modality}/cot"
    if category == "diagnostic_reasoning":
        if source in {"LongTutor-Gold", "LongTutor-Silver", "LongTutor-official"}:
            return "dialogue/long_conversation_history/official"
        if source in {"LongTutor-Synthetic", "LongTutor-synthetic"}:
            return "dialogue/long_conversation_history/synthetic"
        if source == "ScratchMath":
            return "process/multimodal"
        if source == "StepVerify":
            return "process/text"
        if source == "MathDial":
            return "dialogue/no_conversation_history"
        if source == "Bridge":
            return "dialogue/short_conversation_history"
        return "process/text"
    if category == "curriculum_grounding":
        if source == "MathFish-train (multi-relation)":
            return "exercise_standard_mapping/multirelation"
        if source == "MathFish-train (strict)":
            return "exercise_standard_mapping/strict"
        if source.startswith("K12-KGraph"):
            return "knowledge_graph/" + ("multimodal" if "(image)" in source else "text")
        return "general_mapping/multimodal" if source == "Eedi" else "general_mapping/mixed_or_text"
    if kind == "mathtutor_scaffolding":
        return "scaffolding/minimal_scaffold"
    if kind == "oatutor_guidance" or source in {"OATutor", "active-learning-hint"}:
        return "scaffolding/hint_framed_as_a_question"
    if source == "MultiHint":
        return "scaffolding/progressive_hints_no_question"
    if source in {"SocraticMATH", "SocraticMATH-sol", "SocraTeach"}:
        return "questioning/one_question_per_turn"
    if source == "TMATH":
        return "questioning/ordered_question_chain"
    if kind == "tutorbench_dialogue" or source in {"TutorBench", "ConvoLearn"}:
        return "multi_turn_dialogue"
    if source in {"ScratchMath-rewrite", "StepVerify-rewrite", "answer-assessment"}:
        return "assessment/feedback_on_a_solution"
    if source in {"FEAT", "SEFORA", "FOXGLOVE"}:
        return "assessment/feedback_on_writing"
    if source in {"longitudinal-tutor", "LongTutor-dialogue"}:
        return "dialogue/long_conversation_history"
    return "direct_explanation"
