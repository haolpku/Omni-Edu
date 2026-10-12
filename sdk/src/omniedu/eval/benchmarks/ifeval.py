"""IFEval: the official instruction-following protocol.

Protocol source: the ``google/IFEval`` dataset plus Google's scoring code
(``google-research/google-research/ifeval``, Apache-2.0), vendored under
``omniedu/eval/vendor/ifeval_official/`` so the scorer cannot drift.  Generation
is the raw prompt as a single user message, greedy, ``max_tokens=1280``.

Metrics are the official prompt-level and instruction-level strict/loose
accuracies.  The scorer needs ``langdetect``, ``immutabledict`` and ``nltk``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from ..engine import Sink, chat, endpoints, fan_out, marker_failed
from ..io import resume
from ..metrics import build
from ..resources import ifeval_vendor
from ..sampling import Sampling, sampling_from_args, strip_think

NAME = "IFEval"
DEFAULT_MAX_TOKENS = 1280
PROTOCOL = (
    "google/IFEval official: raw prompt, greedy, max_tokens=1280; strict/loose per "
    "google-research/ifeval (vendored via lm-eval v4.0)"
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ifeval-input", default=None, help="ifeval_input_data.jsonl")


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
    source = Path(args.ifeval_input or Path(args.data_root) / "ifeval" / "ifeval_input_data.jsonl")
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = output_dir / "responses.jsonl"
    done = resume(responses, key="key", failed=lambda row: marker_failed(row.get("response")))
    todo = [row for row in rows if str(row["key"]) not in done]
    print(f"[ifeval] total={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    if todo:
        sink = Sink(responses)
        finished = 0

        async def worker(index: int, row: dict[str, Any]) -> None:
            nonlocal finished
            url = urls[index % len(urls)]
            content, reason = await chat(client, url, _payload(args.model, row["prompt"], sampling))
            record = dict(row)
            record["response"] = content
            record["finish_reason"] = reason
            await sink.write(record)
            finished += 1
            if finished % 50 == 0:
                print(f"[ifeval] {finished}/{len(todo)}", flush=True)

        async with httpx.AsyncClient(timeout=600) as client:
            await fan_out(todo, worker, concurrency=args.concurrency)

    return _score(responses, args.model, sampling)


def _load_scorer():
    vendor = str(ifeval_vendor())
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    import instructions_registry  # type: ignore[import-not-found]

    return instructions_registry


def _loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def _score(responses: Path, model: str, sampling: Sampling) -> dict[str, Any]:
    registry = _load_scorer()

    def check(row: dict[str, Any], *, loose: bool) -> list[bool]:
        final = strip_think(row["response"], enabled=sampling.strip_thinking)
        results = []
        for index, instruction_id in enumerate(row["instruction_id_list"]):
            instruction = registry.INSTRUCTION_DICT[instruction_id](instruction_id)
            kwargs = {
                key: value
                for key, value in (row["kwargs"][index] or {}).items()
                if value is not None
            }
            instruction.build_description(**kwargs)
            args_ = instruction.get_instruction_args()
            if args_ and "prompt" in args_:
                instruction.build_description(prompt=row["prompt"])
            if loose:
                ok = any(
                    variant.strip() and instruction.check_following(variant)
                    for variant in _loose_variants(final)
                )
            else:
                ok = bool(final.strip()) and instruction.check_following(final)
            results.append(bool(ok))
        return results

    rows = [
        json.loads(line)
        for line in responses.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: row["key"])
    errors = sum(1 for row in rows if marker_failed(row.get("response")))
    strict_prompt, loose_prompt, strict_inst, loose_inst = [], [], [], []
    for row in rows:
        strict = check(row, loose=False)
        loose = check(row, loose=True)
        strict_prompt.append(all(strict))
        loose_prompt.append(all(loose))
        strict_inst.extend(strict)
        loose_inst.extend(loose)

    summary: dict[str, Any] = {
        "benchmark": "IFEval",
        "model": model,
        "count": len(rows),
        "errors": errors,
        "prompt_level_strict_acc": round(sum(strict_prompt) / len(strict_prompt) * 100, 2),
        "prompt_level_loose_acc": round(sum(loose_prompt) / len(loose_prompt) * 100, 2),
        "inst_level_strict_acc": round(sum(strict_inst) / len(strict_inst) * 100, 2),
        "inst_level_loose_acc": round(sum(loose_inst) / len(loose_inst) * 100, 2),
        "finish_reason_length": sum(1 for row in rows if row.get("finish_reason") == "length"),
        "truncated_no_think": sum(
            1
            for row in rows
            if sampling.strip_thinking and "</think>" not in str(row.get("response", ""))
        ),
        "protocol": PROTOCOL,
    }
    sampling.attach(summary)
    summary["metrics"] = build(
        "IFEval",
        {
            "Prompt Strict": summary["prompt_level_strict_acc"],
            "Inst. Strict": summary["inst_level_strict_acc"],
            "Prompt Loose": summary["prompt_level_loose_acc"],
            "Inst. Loose": summary["inst_level_loose_acc"],
        },
    )
    return summary
