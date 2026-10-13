#!/usr/bin/env python3
"""Dataset to prompt, checks and pool path."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
STAGE2 = HERE.parent / "stage2_deterministic_cleaning_and_evaluation_decontamination"
CAPABILITIES = ("subject_competence", "curriculum_grounding", "diagnostic_reasoning", "pedagogical_action")

PROMPT_OF = {
    "AAAS": "aaas_review",
    "LongTutor-Gold": "longtutor_review",
    "LongTutor-Silver": "longtutor_review",
    "LongTutor-Synthetic": "longtutor_review",
    "Bridge": "bridge_review",
    "MathDial": "mathdial_review",
    "StepVerify": "stepverify_review",
    "ScratchMath": "scratchmath_review",
    "DrawEduMath": "drawedumath_review",
    "CIMA": "cima_review",
    "TalkMoves": "talkmoves_review",
    "MultiHint": "multihint_review",
    "SocraticMATH": "socraticmath_review",
    "SocraticMATH-sol": "socraticmath_review",
    "EduAdapt": "eduadapt_review",
    "TMATH": "tmath_review",
    "LearningQ": "learningq_review",
    "MRBench": "mrbench_review",
    "OATutor": "oatutor_review",
    "TutorBench": "dialogue_general_review",
    "FEAT": "feedback_review",
    "FOXGLOVE": "feedback_review",
    "SEFORA": "feedback_review",
    "CLC-FCE": "clc_fce_review",
    "TutorChat": "education_dialogue_review",
    "Education-Dialogue-Dataset": "education_dialogue_review",
    "ConvoLearn": "dialogue_general_review",
    "SocraTeach": "dialogue_general_review",
    "WikiHint": "semantic_quality_review",
    "algebra_misconceptions": "algebra_misconceptions_review",
    "TalkMoves-teacher": "talkmoves_teacher_review",
    "Eedi": "curriculum_review",
    "MathFish-train (strict)": "curriculum_review",
    "MathFish-train (multi-relation)": "curriculum_review",
    "K12-KGraph (text)": "curriculum_review",
    "K12-KGraph (image)": "curriculum_review",
    "DA-20K": "curriculum_review",
    "TAL-SCQ5K-CN": "curriculum_review",
    "TAL-SCQ5K-EN": "curriculum_review",
    "XES3G5M": "curriculum_review",
    "RACE": "short_answer_5way_review",
    "C3": "short_answer_5way_review",
    "ARC": "short_answer_5way_review",
    "QASC": "short_answer_5way_review",
    "SciQ": "short_answer_5way_review",
    "AI2D": "short_answer_5way_review",
    "ScienceQA (text)": "short_answer_5way_review",
    "ScienceQA (image)": "short_answer_5way_review",
    "Geometry3K": "short_answer_5way_review",
    "GSM8K": "semantic_quality_review",
    "MATH": "semantic_quality_review",
    "MathQA": "semantic_quality_review",
    "OpenR1": "semantic_quality_review",
    "SciInstruct (CN math)": "semantic_quality_review",
    "SciInstruct (EN)": "semantic_quality_review",
    "CJEval": "semantic_quality_review",
    "CMM-Math (text cot)": "semantic_quality_review",
    "CMM-Math (text answer)": "semantic_quality_review",
    "CMM-Math (image)": "semantic_quality_review",
    "TQA (text)": "semantic_quality_review",
    "TQA (diagram)": "semantic_quality_review",
    "WorldTree": "semantic_quality_review",
    "ASAP 2.0": "feedback_review",
    "CSEE": "feedback_review",
    "ELLIPSE": "feedback_review",
}

GENERATE_OF = {
    "LongTutor-Synthetic": "longtutor_review",
    "Eedi-rationale": "eedi_rationale",
    "Eedi-classification": "eedi_classification_rationale",
}

CRITICAL_OF = {
    "reading": ("answer_correct", "grounded_in_passage", "supports_answer"),
    "essay": ("consistent_with_scores", "grounded_in_essay"),
    "qa": ("answer_correct", "cot_supports_answer"),
}


def pool_path(dataset: str) -> Path:
    for capability in CAPABILITIES:
        candidate = STAGE2 / capability / dataset / "cleaned" / "kept.jsonl"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no stage2 pool for dataset {dataset!r}")


def audit_dir(dataset: str) -> Path:
    return pool_path(dataset).parent.parent


def prompt_text(dataset: str) -> str:
    name = PROMPT_OF.get(dataset, "semantic_quality_review")
    path = PROMPTS / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"missing prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


DIMENSION_RE = re.compile(r'"([a-z_0-9]+)"\s*:\s*"yes\|no"')
META_KEYS = {"fatal_issue"}


def dimensions(dataset: str) -> tuple[str, ...]:
    found = tuple(
        key for key in dict.fromkeys(DIMENSION_RE.findall(prompt_text(dataset))) if key not in META_KEYS
    )
    if not found:
        raise ValueError(f"prompt declares no yes/no dimensions: {dataset}")
    return found


def system_prompt(dataset: str, record: dict) -> str:
    text = prompt_text(dataset)
    if "{" in text:
        text = (
            text.replace("{record_json}", json.dumps(record, ensure_ascii=False, sort_keys=True))
            .replace("{dataset}", dataset)
            .replace("{criteria}", "")
        )
    return text
