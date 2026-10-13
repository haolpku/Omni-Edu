#!/usr/bin/env python3
"""Task rubric dimensions, critical dimensions and thresholds."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "sr_1_problem_solving": (
        "problem_validity",
        "answer_correctness",
        "reasoning_correctness",
        "completeness",
        "relevance",
        "clarity",
    ),
    "sr_2_writing_feedback": (
        "task_and_score_validity",
        "score_alignment",
        "feedback_grounding",
        "diagnostic_specificity",
        "actionability_and_completeness",
        "relevance_and_clarity",
    ),
    "sr_5_error_diagnosis": (
        "input_validity",
        "diagnosis_correctness",
        "evidence_grounding",
        "correction_and_content_correctness",
        "pedagogical_quality",
        "context_consistency_and_completeness",
        "clarity_and_efficiency",
    ),
    "sr_6_longitudinal_tutoring": (
        "history_and_input_validity",
        "memory_correctness",
        "diagnosis_evidence",
        "strategy_alignment",
        "tutoring_correctness",
        "personalization_and_pedagogy",
        "answer_leakage_and_efficiency",
        "output_consistency_and_clarity",
    ),
    "sr_3_curriculum_label": (
        "input_validity",
        "label_or_relation_correctness",
        "hierarchy_and_framework_consistency",
        "granularity_and_coverage",
        "grounding_and_explanation_quality",
        "instruction_compliance_and_clarity",
    ),
    "sr_4_strict_standard_alignment": (
        "pair_validity",
        "binary_label_correctness",
        "direct_requirement_precision",
        "standard_specificity",
        "instruction_and_format_compliance",
    ),
    "sr_7_general_pedagogical_action": (
        "input_and_task_validity",
        "correctness_and_grounding",
        "pedagogical_appropriateness",
        "learner_state_adaptation",
        "guidance_and_engagement",
        "completeness_and_instruction_compliance",
        "clarity_and_efficiency",
    ),
    "sr_8_minimal_scaffold": (
        "problem_and_attempt_validity",
        "error_diagnosis",
        "mathematical_correctness",
        "scaffolding_quality",
        "non_leakage",
        "targeting_and_relevance",
        "style_and_efficiency",
    ),
    "sr_9_multi_turn_dialogue": (
        "context_and_input_validity",
        "correctness",
        "relevance",
        "diagnosis_and_feedback",
        "pedagogy_and_clarity",
        "personalization_and_continuity",
        "engagement_and_efficiency",
    ),
    "sr_10_hint_based_guidance": (
        "input_validity",
        "correctness",
        "hint_grounding",
        "next_step_quality",
        "non_leakage",
        "pedagogical_actionability",
        "clarity_and_efficiency",
    ),
}

CRITICAL: dict[str, tuple[str, ...]] = {
    "sr_1_problem_solving": ("problem_validity", "answer_correctness", "reasoning_correctness"),
    "sr_2_writing_feedback": ("task_and_score_validity", "score_alignment", "feedback_grounding"),
    "sr_5_error_diagnosis": (
        "input_validity",
        "diagnosis_correctness",
        "evidence_grounding",
        "correction_and_content_correctness",
    ),
    "sr_6_longitudinal_tutoring": (
        "history_and_input_validity",
        "memory_correctness",
        "diagnosis_evidence",
        "strategy_alignment",
        "tutoring_correctness",
    ),
    "sr_3_curriculum_label": (
        "input_validity",
        "label_or_relation_correctness",
        "hierarchy_and_framework_consistency",
    ),
    "sr_4_strict_standard_alignment": ("pair_validity", "binary_label_correctness"),
    "sr_7_general_pedagogical_action": (
        "input_and_task_validity",
        "correctness_and_grounding",
        "pedagogical_appropriateness",
    ),
    "sr_8_minimal_scaffold": (
        "problem_and_attempt_validity",
        "mathematical_correctness",
        "non_leakage",
    ),
    "sr_9_multi_turn_dialogue": (
        "context_and_input_validity",
        "correctness",
        "diagnosis_and_feedback",
        "pedagogy_and_clarity",
    ),
    "sr_10_hint_based_guidance": (
        "input_validity",
        "correctness",
        "hint_grounding",
        "next_step_quality",
        "non_leakage",
    ),
}

READING_SOURCES = ("RACE", "C3")
IGNORED_DIMENSIONS = {"problem_validity": READING_SOURCES}

FLOOR = 3
CRITICAL_FLOOR = 4


def prompt_text(family: str) -> str:
    path = PROMPTS / f"{family}.md"
    if not path.is_file():
        raise FileNotFoundError(f"missing rubric prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def system_prompt(family: str, record: dict) -> str:
    import json

    text = prompt_text(family)
    if "{" in text:
        text = text.replace("{record_json}", json.dumps(record, ensure_ascii=False, sort_keys=True))
    return text


def numeric_scores(obj: dict) -> dict:
    raw = obj.get("scores") if isinstance(obj.get("scores"), dict) else obj
    scores: dict[str, float] = {}
    for key, value in raw.items():
        if key in ("reason", "scores", "error"):
            continue
        try:
            scores[key] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def keeps(family: str, scores: dict, source: str = "") -> bool:
    dims = DIMENSIONS.get(family, ())
    values = {
        dim: float(scores[dim])
        for dim in dims
        if dim in scores and scores[dim] not in (None, "")
    }
    for dim, exempt_sources in IGNORED_DIMENSIONS.items():
        if source in exempt_sources:
            values.pop(dim, None)
    if not values:
        return False
    if any(value < FLOOR for value in values.values()):
        return False
    critical = [dim for dim in CRITICAL.get(family, ()) if dim in values]
    return all(values[dim] >= CRITICAL_FLOOR for dim in critical)
