#!/usr/bin/env python3
"""Multiple-choice cleaning: RACE, C3, TQA (text/diagram), WorldTree, AI2D."""

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
SPLITS = ("train", "validation", "test")


def has_image(row: dict) -> bool:
    for key in ("image", "images", "image_bytes", "image_path"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("bytes"):
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def mcq_rows(dataset: str, split: str, rows) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    removed: list[dict] = []
    for index, row in enumerate(rows):
        uid = str(cleaning.pick(row, "example_id", "id", "question_id", default=f"{split}-{index}"))
        problem = str(cleaning.pick(row, "question", "problem", "query")).strip()
        options = cleaning.pick(row, "options", "choices", "answer_choices", default=[])
        mapping = cleaning.option_map(options)
        raw_answer = cleaning.pick(row, "answer", "answerKey", "correct", "label", default="")
        labels = list(mapping)
        if isinstance(raw_answer, int) and 0 < raw_answer <= len(labels):
            answer = labels[raw_answer - 1]
        else:
            answer = cleaning.answer_letter(row, options)
        if not problem or not mapping or str(answer).upper() not in mapping:
            removed.append({"uid": uid, "_reason": "missing_problem_options_or_answer"})
            continue
        content = cleaning.pick(row, "article", "passage", "context", default="")
        passage = "\n".join(str(part) for part in content) if isinstance(content, list) else str(content)
        facts = [
            str(cleaning.pick(row, key, default="")).strip()
            for key in ("combinedfact", "fact1", "fact2")
        ]
        explanation = cleaning.pick(row, "explanation", "facts", default="")
        if isinstance(explanation, list):
            facts.extend(str(item) for item in explanation)
        cot = "\n".join(fact for fact in facts if fact)
        image = has_image(row) or dataset == "AI2D"
        target = f"{dataset} (image)" if dataset == "TQA" and image else dataset
        kept.append(
            cleaning.record(
                target,
                uid,
                problem,
                options=options,
                answer=str(answer).upper(),
                answer_text=mapping.get(str(answer).upper(), ""),
                cot=cot,
                passage=passage.strip(),
                meta={"split": split, "lang": "zh" if dataset == "C3" else "en"},
                image_ref={"origin_index": index, "has_image": True} if image else None,
                flags=["multimodal_deferred"] if image else (),
            )
        )
    return kept, removed


def c3_rows(split: str, documents) -> tuple[list[dict], list[dict]]:
    kept: list[dict] = []
    removed: list[dict] = []
    for doc_index, document in enumerate(documents):
        title = str(cleaning.pick(document, "title", default="")).strip()
        content = cleaning.pick(document, "content", "passage", default="")
        passage = "\n".join(str(part) for part in content) if isinstance(content, list) else str(content)
        for question_index, question in enumerate(cleaning.pick(document, "questions", default=[])):
            uid = f"{split}-{doc_index}-{question_index}"
            problem = str(cleaning.pick(question, "question", "query", default="")).strip()
            options = cleaning.pick(question, "choices", "options", default={})
            mapping = cleaning.option_map(options)
            answer_text = str(cleaning.pick(question, "answer", "answer_text", default="")).strip()
            answer = next((label for label, text in mapping.items() if text.strip() == answer_text), "")
            if not problem or not mapping or not answer:
                removed.append({"uid": uid, "_reason": "missing_problem_options_or_answer"})
                continue
            kept.append(
                cleaning.record(
                    "C3",
                    uid,
                    problem,
                    options=options,
                    answer=answer,
                    answer_text=answer_text,
                    passage=f"{title}\n{passage}".strip(),
                    meta={"split": split, "lang": "zh"},
                )
            )
    return kept, removed


def load_split(source: Path, split: str):
    split_dir = source / split
    target = split_dir if split_dir.is_dir() else source
    files = [path for path in sorted(target.glob("*")) if path.is_file()]
    rows = []
    for path in files:
        if path.suffix in {".parquet", ".jsonl", ".json"}:
            rows.append((path, list(cleaning.iter_rows(path))))
    return rows


def run_dataset(dataset: str, source: Path) -> dict[str, dict]:
    pools_out: dict[str, list[dict]] = {}
    removed: dict[str, list[dict]] = {}
    for split in SPLITS:
        for path, rows in load_split(source, split):
            if dataset == "C3":
                split_kept, split_removed = c3_rows(split if split != "train" else path.stem, rows)
                key = "C3"
            elif dataset == "TQA":
                split_kept, split_removed = mcq_rows("TQA", split, rows)
                key = None
            else:
                split_kept, split_removed = mcq_rows(dataset, split, rows)
                key = dataset
            if key is None:
                for row in split_kept:
                    pools_out.setdefault(row["dataset"], []).append(row)
            else:
                pools_out.setdefault(key, []).extend(split_kept)
            if split_removed:
                removed.setdefault(split, []).extend(split_removed)
    return {
        name: cleaning.write_pool(CAPABILITY, name, rows, removed=removed if name == next(iter(pools_out)) else {})
        for name, rows in pools_out.items()
    }


SOURCES = {
    "RACE": SOURCE_ROOT / "huggingface" / "race",
    "C3": SOURCE_ROOT / "github" / "c3",
    "TQA": SOURCE_ROOT / "huggingface" / "tqa",
    "WorldTree": SOURCE_ROOT / "huggingface" / "worldtree",
    "AI2D": SOURCE_ROOT / "huggingface" / "ai2d",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), default=None, help="clean one source only")
    args = parser.parse_args()

    for dataset in sorted(SOURCES):
        if args.dataset and dataset != args.dataset:
            continue
        source = SOURCES[dataset]
        if not source.exists():
            print({"source": dataset, "skipped": f"missing snapshot: {source}"})
            continue
        print(run_dataset(dataset, source))


if __name__ == "__main__":
    main()
