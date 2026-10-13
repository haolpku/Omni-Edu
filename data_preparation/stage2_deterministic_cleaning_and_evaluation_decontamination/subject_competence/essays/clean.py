#!/usr/bin/env python3
"""Essay cleaning: ASAP 2.0, CSEE, ELLIPSE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import cleaning, pools

CAPABILITY = "subject_competence"
SOURCE_ROOT = pools.source_dir(CAPABILITY)

SOURCES = {
    "ASAP 2.0": SOURCE_ROOT / "github" / "ASAP_2.0",
    "CSEE": SOURCE_ROOT / "huggingface" / "Chinese-Student-English-Essay",
    "ELLIPSE": SOURCE_ROOT / "huggingface" / "ELLIPSE",
}


def prompt_text(row: dict) -> str:
    prompt = cleaning.pick(row, "prompt", "assignment", "prompt_name", "title", "text_id", default="")
    if isinstance(prompt, dict):
        for key in ("prompt", "text", "name"):
            if prompt.get(key):
                return str(prompt[key]).strip()
    return str(prompt).strip()


def essay_text(row: dict) -> str:
    return str(cleaning.pick(row, "essay", "full_text", "text", "student_essay", default="")).strip()


def scores_of(row: dict):
    for key in ("scores", "score", "annotations", "domain1_score", "overall_score"):
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return ""


def feedback_of(row: dict) -> str:
    value = cleaning.pick(row, "feedback", "comment", "annotations_text", default="")
    return str(value).strip()


def build(dataset: str, source: Path) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    removed: list[dict] = []
    for index, (path, row) in enumerate(cleaning.iter_snapshot(source)):
        essay = essay_text(row)
        prompt = prompt_text(row)
        if not essay:
            removed.append(
                {"uid": str(cleaning.pick(row, "essay_id", "id", default=f"{index}")),
                 "_reason": "empty_essay", "source_file": path.name}
            )
            continue
        kept.append(
            {
                "uid": f"{dataset}:{cleaning.pick(row, 'essay_id', 'id', default=path.stem + '-' + str(index))}",
                "dataset": dataset,
                "prompt": prompt,
                "essay": essay,
                "scores": scores_of(row),
                "feedback": feedback_of(row),
                "meta": {
                    "split": str(cleaning.pick(row, "split", "set", default="unsplit")),
                    "lang": "zh" if dataset == "CSEE" else "en",
                    "subject": "writing",
                    "source_file": path.name,
                },
            }
        )
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), default=None, help="clean one dataset only")
    args = parser.parse_args()

    for dataset in sorted(SOURCES):
        if args.dataset and dataset != args.dataset:
            continue
        source = SOURCES[dataset]
        if not source.exists():
            print({"dataset": dataset, "skipped": f"missing source snapshot: {source}"})
            continue
        kept, removed = build(dataset, source)
        print(cleaning.write_pool(CAPABILITY, dataset, kept, removed={"invalid": removed}))


if __name__ == "__main__":
    main()
