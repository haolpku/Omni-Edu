"""GPQA: the official two-stage closed-book chain-of-thought protocol.

Port of ``github.com/idavidrein/gpqa`` ``baselines/{closed_book,utils}.py``:
CSV rows in order with ``random.seed(0)`` and a per-row choice shuffle; a base
prompt plus "Let's think step by step:" for the reasoning stage, then the same
prompt followed by the model's own reasoning and "Based on the above ..." for
the answer stage; the official regex chain parses the answer and ``question_id
== 69`` is counted wrong without a call (the official hack).

Generation is greedy with ``max_tokens=1000`` (the official default).  GPQA's
CSV files are gated, so they are not redistributed; place ``gpqa_main.csv`` and
``gpqa_diamond.csv`` under ``<data-root>/gpqa/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import namedtuple
from pathlib import Path
from typing import Any

import pandas as pd

from ..engine import Sink, chat, endpoints, fan_out, marker_failed
from ..io import resume
from ..metrics import build
from ..sampling import Sampling, sampling_from_args, strip_think

NAME = "GPQA"
DEFAULT_MAX_TOKENS = 1000

Example = namedtuple(
    "Example", ["question", "choice1", "choice2", "choice3", "choice4", "correct_index"]
)
LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}
PARSE_PATTERNS = [
    r"answer is \((.)\)",
    r"Answer: \((.)\)",
    r"answer: \((.)\)",
    r"answer \((.)\)",
    r"\((.)\)",
]
PROTOCOL = (
    "Idavidrein/gpqa official closed_book zero_shot_chain_of_thought: seed=0 choice "
    "shuffle, two-stage CoT, temp=0, max_tokens=1000, official regex parse, qid69 "
    "counted wrong"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpqa-data-dir", default=None, help="directory holding gpqa_main.csv")


def load_examples(path: Path, *, seed: int = 0) -> list[Example]:
    frame = pd.read_csv(path)
    random.seed(seed)

    def shuffle(row: Any) -> Example:
        choices = [
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
            row["Correct Answer"],
        ]
        random.shuffle(choices)
        return Example(
            row.Question,
            choices[0],
            choices[1],
            choices[2],
            choices[3],
            choices.index(row["Correct Answer"]),
        )

    return [shuffle(row) for _, row in frame.iterrows()]


def base_prompt(example: Example) -> str:
    prompt = f"What is the correct answer to this question: {example.question}"
    return (
        prompt
        + f"\n\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}"
        + f"\n(C) {example.choice3}\n(D) {example.choice4}"
    )


def parse_sampled_answer(answer: str) -> str | None:
    for pattern in PARSE_PATTERNS:
        match = re.search(pattern, answer)
        if match and match.group(1) in LETTER_TO_INDEX:
            return match.group(1)
    return None


def _payload(model: str, prompt: str, sampling: Sampling) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": sampling.temperature,
        "max_tokens": sampling.max_tokens,
    }
    sampling.apply(payload)
    if sampling.chat_template_kwargs:
        payload["chat_template_kwargs"] = sampling.chat_template_kwargs
    return payload


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    sampling = sampling_from_args(args, default_max_tokens=DEFAULT_MAX_TOKENS)
    urls = endpoints(args.base_urls)
    data_dir = Path(args.gpqa_data_dir or Path(args.data_root) / "gpqa")
    examples = load_examples(data_dir / "gpqa_main.csv", seed=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "generations.jsonl"
    done = resume(
        cache,
        key="qid",
        failed=lambda row: marker_failed(row.get("stage1")) or marker_failed(row.get("stage2")),
    )
    todo = [index for index in range(len(examples)) if str(index) not in done]
    print(f"[gpqa] total={len(examples)} done={len(done)} todo={len(todo)}", flush=True)

    sink = Sink(cache)
    finished = 0

    async def worker(index: int, qid: int) -> None:
        nonlocal finished
        example = examples[qid]
        if qid == 69:  # official hack: choices too long, counted wrong without a call
            record: dict[str, Any] = {
                "qid": qid,
                "question": example.question,
                "correct_index": example.correct_index,
                "stage1": None,
                "stage2": None,
                "pred": None,
                "skipped_long_choices": True,
            }
        else:
            url = urls[qid % len(urls)]
            stage1_prompt = base_prompt(example) + "\nLet's think step by step: "
            reasoning, reason1 = await chat(
                client, url, _payload(args.model, stage1_prompt, sampling)
            )
            stage2_prompt = (
                stage1_prompt
                + f"{reasoning}\n\nBased on the above, what is the single, most likely answer "
                'choice? Answer in the format "The correct answer is (insert answer here)".'
            )
            answer, reason2 = await chat(client, url, _payload(args.model, stage2_prompt, sampling))
            record = {
                "qid": qid,
                "question": example.question,
                "correct_index": example.correct_index,
                "stage1": reasoning,
                "stage2": answer,
                "stage1_finish_reason": reason1,
                "stage2_finish_reason": reason2,
                "pred": parse_sampled_answer(strip_think(answer, enabled=sampling.strip_thinking)),
            }
        await sink.write(record)
        finished += 1
        if finished % 25 == 0:
            print(f"[gpqa] {finished}/{len(todo)}", flush=True)

    async with httpx.AsyncClient(timeout=600) as client:
        await fan_out(todo, worker, concurrency=args.concurrency)

    return _score(cache, examples, data_dir, args.model, sampling)


def _score(
    cache: Path, examples: list[Example], data_dir: Path, model: str, sampling: Sampling
) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in cache.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    rows.sort(key=lambda row: row["qid"])
    assert len(rows) == len(examples), f"{len(rows)} != {len(examples)}"

    correct = sum(
        row["pred"] is not None and LETTER_TO_INDEX[row["pred"]] == row["correct_index"]
        for row in rows
    )
    refusals = sum(row["pred"] is None for row in rows)
    result: dict[str, Any] = {
        "benchmark": "GPQA",
        "subset": "gpqa_main",
        "model": model,
        "count": len(rows),
        "accuracy": round(correct / len(rows) * 100, 2),
        "refusals": refusals,
    }

    diamond_path = data_dir / "gpqa_diamond.csv"
    if diamond_path.exists():
        diamond_questions = set(pd.read_csv(diamond_path)["Question"].tolist())
        subset = [row for row in rows if row["question"] in diamond_questions]
        hit = sum(
            row["pred"] is not None and LETTER_TO_INDEX[row["pred"]] == row["correct_index"]
            for row in subset
        )
        result["diamond_count"] = len(subset)
        result["diamond_accuracy"] = round(hit / max(len(subset), 1) * 100, 2)

    result["stage1_finish_reason_length"] = sum(
        1 for row in rows if row.get("stage1_finish_reason") == "length"
    )
    result["stage2_finish_reason_length"] = sum(
        1 for row in rows if row.get("stage2_finish_reason") == "length"
    )
    result["stage2_truncated_no_think"] = sum(
        1
        for row in rows
        if sampling.strip_thinking and "</think>" not in str(row.get("stage2") or "")
    )
    result["protocol"] = PROTOCOL
    sampling.attach(result)
    result["metrics"] = build(
        "GPQA",
        {"Main Acc.": result["accuracy"], "Diamond Acc.": result.get("diamond_accuracy")},
    )
    return result
