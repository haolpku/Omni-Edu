#!/usr/bin/env python3
"""Regenerate ``src/omniedu/tasks.json`` from the research repository.

The 19 pedagogical instructions live in exactly one place in this repository:
``data_preparation/stage6_.../assign_system_prompts.py``. The released package
needs its own copy so it can be installed standalone, so this script copies them
across and records a content hash. Run it whenever the stage-6 instructions
change, then re-run ``pytest`` (``tests/test_tasks.py`` checks the two agree).

    python sdk/tools/sync_tasks.py            # write tasks.json
    python sdk/tools/sync_tasks.py --check    # fail if tasks.json is stale
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SDK_ROOT.parent
STAGE6 = (
    REPO_ROOT
    / "data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly"
)
SOURCE = STAGE6 / "assign_system_prompts.py"
TARGET = SDK_ROOT / "src" / "omniedu" / "tasks.json"

#: Task id -> category, so the shipped registry is browsable. Ids come from the
#: source file; a new id without a category here is a hard error, which is what
#: we want when stage 6 grows.
CATEGORIES: dict[str, str] = {
    "solve_answer_only": "solving",
    "solve_reasoned": "solving",
    "reading_comprehension": "solving",
    "diagnose_and_correct": "diagnosis",
    "answer_assessment": "diagnosis",
    "writing_feedback": "diagnosis",
    "knowledge_point": "curriculum",
    "mathfish_strict": "curriculum",
    "mathfish_multirelation": "curriculum",
    "mathtutor_scaffolding": "tutoring",
    "mathtutor_pedagogy_following": "tutoring",
    "tutorbench": "tutoring",
    "active_probe": "tutoring",
    "multihint": "tutoring",
    "multiturn_socratic": "tutoring",
    "oatutor_guidance": "tutoring",
    "socratic_question_chain": "tutoring",
    "direct_explanation": "tutoring",
    "longtutor_official": "tutoring",
}


def load_stage6_module():
    if not SOURCE.is_file():
        raise SystemExit(
            f"stage-6 source not found at {SOURCE}. This tool only runs inside the "
            "Omni-Edu research repository."
        )
    spec = importlib.util.spec_from_file_location("_omniedu_stage6", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_payload() -> dict:
    module = load_stage6_module()
    registry: dict[str, str] = dict(module.PROMPTS)

    unknown = sorted(set(registry) - set(CATEGORIES))
    if unknown:
        raise SystemExit(
            f"new task ids without a category: {unknown}. Add them to CATEGORIES "
            "in sdk/tools/sync_tasks.py and re-run."
        )

    tasks = {
        name: {
            "instruction": text,
            "category": CATEGORIES[name],
            "source": "data_preparation/stage6_pedagogical_instruction_assignment_and_final_assembly",
        }
        for name, text in registry.items()
    }
    digest = hashlib.sha256(
        json.dumps(tasks, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "version": digest[:12],
        "source": str(SOURCE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "tasks": tasks,
    }


def render(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when tasks.json does not match the stage-6 source",
    )
    args = parser.parse_args(argv)

    text = render(build_payload())
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current != text:
            print(f"{TARGET} is out of date; run sdk/tools/sync_tasks.py", file=sys.stderr)
            return 1
        print(f"{TARGET} is up to date")
        return 0

    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET} ({len(json.loads(text)['tasks'])} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
