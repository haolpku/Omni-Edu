"""MathFish verification: replay the upstream prompt and write its prediction file.

Alignment with upstream (``mathfish/modelWrappers/verification_wrapper.py`` and
``scripts/run_wrapper.py``): the same ``prompt_template.format(...)`` call,
including the branch that uses three format arguments when a row has
``few_shots``; optional token-based truncation mirroring ``truncate_messages``;
and no ``max_tokens`` (the upstream ``call_api`` sends none).

Scoring is the official ``<clone>/scripts/eval.py --task_format verification``,
which reads exactly the four keys this runner writes: ``id``, ``dataset``,
``message_content`` and ``model-response``.  The accuracy reported here is
informational; the official scorer is authoritative.

One deliberate deviation: ``temperature=0.0`` follows the paper's protocol.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..engine import THINKING_OFF, Sink, chat, endpoints, fan_out
from ..io import resume
from ..metrics import build

NAME = "MathFish"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mathfish-input", default=None, help="data_verification-*.jsonl")
    parser.add_argument(
        "--truncate", action="store_true", help="enable upstream token truncation (default off)"
    )
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer used for --truncate")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--buffer", type=int, default=10)


def format_prompt(user: dict[str, Any]) -> str:
    """Official ``VerificationWrapper.format_prompt`` (2 args without few_shots)."""
    if user.get("few_shots"):
        return user["prompt_template"].format(
            problem_activity=user["problem_activity"],
            standard_description=user["standard_description"],
            few_shots=user["few_shots"],
        )
    return user["prompt_template"].format(
        problem_activity=user["problem_activity"],
        standard_description=user["standard_description"],
    )


def truncate_activity(
    user: dict[str, Any], tokenizer: Any, max_length: int, buffer: int
) -> tuple[dict[str, Any], int, bool]:
    """Official ``VerificationWrapper.truncate_messages`` (token based)."""

    def length(text: Any) -> int:
        return tokenizer(text, truncation=False, return_length=True, add_special_tokens=False)[
            "length"
        ][0]

    template = length(user.get("prompt_template"))
    description = length(user.get("standard_description"))
    activity = tokenizer(user.get("problem_activity"), truncation=False, add_special_tokens=False)
    current = template + description + activity["length"][0]
    truncated = False
    if current > (max_length - buffer):
        room = max_length - template - description - buffer
        if room < 0:
            raise ValueError("prompt template and standard_description do not fit in max_length")
        activity["input_ids"] = activity["input_ids"][:room]
        user = dict(user)
        user["problem_activity"] = tokenizer.decode(activity["input_ids"])
        truncated = True
    return user, current, truncated


def parse_yes_no(text: str) -> str:
    """Official ``VerificationWrapper.verify_response`` parser."""
    for token in re.split(r"\. |\s+|\n", text or ""):
        token = re.sub(r"[.,\s]", "", token.lower())
        if token in {"yes", "no"}:
            return token
    return ""


def _payload(model: str, prompt: str) -> dict[str, Any]:
    # Upstream call_api sends model + messages + temperature only (no max_tokens).
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "chat_template_kwargs": THINKING_OFF,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import httpx

    urls = endpoints(args.base_urls)
    source = args.mathfish_input
    if not source:
        candidates = sorted((Path(args.data_root) / "mathfish").glob("data_verification-*.jsonl"))
        if not candidates:
            raise SystemExit(
                f"no data_verification-*.jsonl under {Path(args.data_root) / 'mathfish'}"
            )
        source = candidates[0]
    rows = [
        json.loads(line)
        for line in Path(source).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    tokenizer = None
    max_length = args.max_length
    if args.truncate:
        if not args.tokenizer:
            raise SystemExit("--truncate requires --tokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if max_length is None:
            limit = getattr(tokenizer, "model_max_length", None)
            max_length = limit if isinstance(limit, int) and limit < 10**30 else 4097

    output = Path(args.output_dir) / f"{Path(source).stem}_model-{args.model}_response.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = resume(output, key="id", failed=lambda row: bool(row.get("error")))
    todo = [row for row in rows if row["id"] not in done]
    print(f"[mathfish] total={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    sink = Sink(output)
    completed = 0

    async def worker(index: int, row: dict[str, Any]) -> None:
        nonlocal completed
        user = dict(row["messages"][0])
        truncated = False
        tokens = None
        if tokenizer is not None:
            user, tokens, truncated = truncate_activity(user, tokenizer, max_length, args.buffer)
        prompt = format_prompt(user)
        url = urls[index % len(urls)]
        response, _ = await chat(client, url, _payload(args.model, prompt))
        record = {
            "dataset": row["dataset"],
            "id": row["id"],
            "message_content": prompt,
            "model-response": response,
            "parsed_prediction": parse_yes_no(response),
            "gold": row["messages"][-1]["aligns"],
            "model": args.model,
            "truncate": bool(args.truncate),
            "truncated": truncated,
            "prompt_tokens": tokens,
        }
        if response.startswith("__ERROR__"):
            record["error"] = response
        await sink.write(record)
        completed += 1
        if (len(done) + completed) % 250 == 0:
            print(f"[mathfish] {len(done) + completed}/{len(rows)}", flush=True)

    async with httpx.AsyncClient(timeout=300) as client:
        await fan_out(
            todo, worker, concurrency=args.concurrency, chunk=max(64, args.concurrency * 4)
        )

    return _summarize(output, args.model, rows, max_length, args)


def _summarize(
    output: Path,
    model: str,
    rows: list[dict[str, Any]],
    max_length: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    values = {}
    for line in output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            values[record["id"]] = record
        except json.JSONDecodeError:
            continue
    items = list(values.values())
    total = len(items) or 1
    correct = sum(item["parsed_prediction"] == item["gold"] for item in items)
    summary = {
        "benchmark": "MathFish-verification",
        "model": model,
        "count": len(items),
        "correct": correct,
        "accuracy": correct / total,
        "malformed": sum(item["parsed_prediction"] == "" for item in items),
        "truncated": sum(bool(item.get("truncated")) for item in items),
        "errors": sum(bool(item.get("error")) for item in items),
        "temperature": 0.0,
        "max_tokens": "not sent (upstream call_api sends none)",
        "truncate": bool(args.truncate),
        "max_length": max_length,
        "buffer": args.buffer,
        "note": "accuracy is informational; the official <clone>/scripts/eval.py --task_format verification is authoritative",
    }
    summary["metrics"] = _metrics(output, accuracy=correct / total, args=args)
    return summary


def _pct(value: Any) -> float | None:
    return None if value is None else float(value) * 100


def _metrics(output: Path, accuracy: float, args: argparse.Namespace) -> dict[str, Any]:
    """Prefer the official scorer's breakdown; fall back to our informational parse."""
    official = _official(output, args)
    values: dict[str, float | None] = {}
    if official:
        values = {
            "Pos: Should Yes": _pct(official.get("acc_pos")),
            "Neg: ATC Neighbor": _pct(official.get("acc_neighbors")),
            "Neg: Same-domain Same-grade": _pct(official.get("acc_same-domain-same-grade")),
            "Neg: Same-domain Diff-grade": _pct(official.get("acc_same-domain-different-grade")),
            "Neg: Diff-domain Same-grade": _pct(official.get("acc_different-domain-same-grade")),
            "Neg: Diff-domain Diff-grade": _pct(
                official.get("acc_different-domain-different-grade")
            ),
            "Overall Acc": _pct(official.get("acc")),
        }
    if values.get("Overall Acc") is None:
        values["Overall Acc"] = accuracy * 100
    return build("MathFish", values)


def _official(output: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    """Run ``<clone>/scripts/eval.py --task_format verification`` when available."""
    import os
    import subprocess

    repo = getattr(args, "official", {}).get(NAME)
    source = getattr(args, "mathfish_input", None)
    if repo is None or not source:
        return None
    repo = Path(repo)
    script = repo / "scripts" / "eval.py"
    data_root = Path(args.data_root) / "mathfish"
    standards, problems = data_root / "standards.jsonl", data_root / "dev.jsonl"
    if not (script.is_file() and standards.is_file() and problems.is_file()):
        return None
    scored = output.parent / "official_eval"
    scored.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(args.python),
            "scripts/eval.py",
            "--predictions_file",
            str(output),
            "--standards_file",
            str(standards),
            "--problems_file",
            str(problems),
            "--data_file",
            str(source),
            "--output_path",
            str(scored),
            "--task_format",
            "verification",
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo)},
        text=True,
    )
    produced = sorted(scored.glob("results_*.jsonl"))
    if not produced:
        return None
    for row in produced[-1].read_text(encoding="utf-8").splitlines():
        if row.strip():
            return json.loads(row)
    return None
