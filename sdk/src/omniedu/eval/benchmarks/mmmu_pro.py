"""MMMU-Pro: the official standard 10-option chain-of-thought protocol.

Port of ``MMMU-Benchmark/MMMU`` ``mmmu-pro/infer/infer_gpt.py`` and
``mmmu-pro/evaluate.py``: the question, one ``A. option`` per line, and the
official ``cot`` / ``standard`` instruction; ``<image i>`` tokens are replaced by
``<image>`` and the images are attached in order of appearance; greedy with
``max_tokens=4096``; the official ``parse_multi_choice_response`` and the
domain/subject aggregation over ``DOMAIN_CAT2SUB_CAT``.

MMMU-Pro is a vision benchmark - every row carries images - so a text-only model
should exclude it rather than run it.
"""

from __future__ import annotations

import argparse
import ast
import base64
import glob
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..engine import Sink, chat, endpoints, fan_out, marker_failed
from ..io import resume
from ..metrics import MMMU_DOMAINS, build
from ..sampling import Sampling, sampling_from_args, strip_think

NAME = "MMMU-Pro"
DEFAULT_MAX_TOKENS = 4096
PROTOCOL = (
    "MMMU-Benchmark/MMMU mmmu-pro official: infer_gpt.py prompt (cot/standard), "
    "temp=0, max_tokens=4096, evaluate.py parsing"
)

COT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of options. Think step by step before answering."
)

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mmmu-data-dir", default=None, help='MMMU-Pro "standard (10 options)" parquet shards'
    )


def replace_images_tokens(text: str) -> tuple[str, list[int]]:
    order = [int(number) for number in re.findall(r"<image\s+(\d+)>", text)]
    return re.sub(r"<image\s+\d+>", "<image>", text), order


def construct_prompt(doc: dict[str, Any]) -> str:
    options = ast.literal_eval(str(doc["options"]))
    letters = [chr(ord("A") + index) for index in range(len(options))]
    choices = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return f"{doc['question']}\n{choices}\n{COT_STANDARD}"


def get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    index2ans: dict[str, str] = {}
    all_choices: list[str] = []
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        index2ans[letter] = option
        all_choices.append(letter)
    return index2ans, all_choices


def parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str:
    """Official MMMU parser (``eval_utils.py``)."""
    if not isinstance(response, str) or not response:
        response = ""
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching = [option for option in all_choices if option in answer_str]
        if len(matching) == 1:
            return matching[0]

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, answer in index2ans.items():
            if answer.lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if len(candidates) == 0:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            for candidate in candidates:
                needle = f"({candidate})" if ans_with_brack else f" {candidate} "
                start_indexes.append(response.rfind(needle))
        else:
            for candidate in candidates:
                start_indexes.append(response.lower().rfind(index2ans[candidate].lower()))
        import numpy as np

        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]
    return pred_index


def _encode_png(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _payload(model: str, prompt: str, images: list[str], sampling: Sampling) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for encoded in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
        )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": sampling.temperature,
        "max_tokens": sampling.max_tokens,
    }
    sampling.apply(payload)
    if sampling.chat_template_kwargs:
        payload["chat_template_kwargs"] = sampling.chat_template_kwargs
    return payload


def load_docs(data_dir: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for shard in sorted(glob.glob(str(data_dir / "*.parquet"))):
        docs.extend(pq.read_table(shard).to_pylist())
    return docs


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from PIL import Image

    import httpx

    sampling = sampling_from_args(args, default_max_tokens=DEFAULT_MAX_TOKENS)
    urls = endpoints(args.base_urls)
    data_dir = Path(
        args.mmmu_data_dir or Path(args.data_root) / "mmmu_pro" / "standard (10 options)"
    )
    docs = load_docs(data_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "generations.jsonl"
    done = resume(cache, key="id", failed=lambda row: marker_failed(row.get("response")))
    todo = [doc for doc in docs if doc["id"] not in done]
    print(f"[mmmu_pro] total={len(docs)} done={len(done)} todo={len(todo)}", flush=True)

    sink = Sink(cache)
    finished = 0

    async def worker(index: int, doc: dict[str, Any]) -> None:
        nonlocal finished
        prompt, order = replace_images_tokens(construct_prompt(doc))
        images = []
        for position in order:
            value = doc.get(f"image_{position}")
            if value is not None and value.get("bytes"):
                images.append(_encode_png(Image.open(io.BytesIO(value["bytes"]))))
        url = urls[index % len(urls)]
        content, reason = await chat(client, url, _payload(args.model, prompt, images, sampling))
        await sink.write(
            {
                "id": doc["id"],
                "question": doc["question"],
                "options": doc["options"],
                "answer": doc["answer"],
                "subject": doc["subject"],
                "response": content,
                "finish_reason": reason,
            }
        )
        finished += 1
        if finished % 50 == 0:
            print(f"[mmmu_pro] {finished}/{len(todo)}", flush=True)

    async with httpx.AsyncClient(timeout=900) as client:
        await fan_out(todo, worker, concurrency=args.concurrency)

    return _score(cache, docs, args.model, sampling)


def _score(
    cache: Path, docs: list[dict[str, Any]], model: str, sampling: Sampling
) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == len(docs), f"{len(rows)} != {len(docs)}"
    errors = sum(1 for row in rows if marker_failed(row["response"]))

    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        options = ast.literal_eval(str(row["options"]))
        index2ans, all_choices = get_multi_choice_info(options)
        row["parsed_pred"] = parse_multi_choice_response(
            strip_think(row["response"], enabled=sampling.strip_thinking), all_choices, index2ans
        )
        row["if_right"] = bool(row["parsed_pred"] == row["answer"])
        by_subject[row["subject"]].append(row)

    subject_acc = {
        subject: {
            "num": len(items),
            "acc": round(sum(item["if_right"] for item in items) / len(items), 4),
        }
        for subject, items in by_subject.items()
    }

    def domain_acc(categories: list[str]) -> dict[str, Any]:
        num = sum(subject_acc[cat]["num"] for cat in categories if cat in subject_acc)
        acc = sum(
            subject_acc[cat]["acc"] * subject_acc[cat]["num"]
            for cat in categories
            if cat in subject_acc
        )
        return {"num": num, "acc": round(acc / num, 4) if num else 0.0}

    printable = {
        f"Overall-{domain}": domain_acc(cats) for domain, cats in DOMAIN_CAT2SUB_CAT.items()
    }
    total_num = sum(value["num"] for value in subject_acc.values())
    total_acc = sum(value["acc"] * value["num"] for value in subject_acc.values())
    printable["Overall"] = {
        "num": total_num,
        "acc": round(total_acc / total_num, 4),
    }

    summary: dict[str, Any] = {
        "benchmark": "MMMU-Pro",
        "setting": "standard (10 options)",
        "mode": "cot",
        "model": model,
        "count": total_num,
        "errors": errors,
        "overall_acc": round(total_acc / total_num * 100, 2),
        "domains": {key: round(value["acc"] * 100, 2) for key, value in printable.items()},
        "subjects": {key: round(value["acc"] * 100, 2) for key, value in subject_acc.items()},
        "finish_reason_length": sum(1 for row in rows if row.get("finish_reason") == "length"),
        "truncated_no_think": sum(
            1
            for row in rows
            if sampling.strip_thinking and "</think>" not in str(row.get("response", ""))
        ),
        "protocol": PROTOCOL,
    }
    sampling.attach(summary)
    # ``domains`` is keyed ``Overall-<full domain name>``; the appendix uses an
    # abbreviation per domain.
    values: dict[str, float | None] = {"Overall": summary["overall_acc"]}
    for domain, column in MMMU_DOMAINS.items():
        values[column] = summary["domains"].get(f"Overall-{domain}")
    summary["metrics"] = build("MMMU-Pro", values)
    return summary
