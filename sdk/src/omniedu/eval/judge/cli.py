"""``omniedu judge``: score generated responses with the benchmark's judge.

The task selects which judge runs; the packaged official prompts are filled in
automatically so the call matches the protocol the score is defined by.

    omniedu judge --task gaokao    --input responses.jsonl --output judged.jsonl ...
    omniedu judge --task tutorbench --input predictions.jsonl --output judged.jsonl ...
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from ..resources import judge_prompt

#: The paper's "maximum judge output length" per benchmark (Appendix: *LLM-based
#: evaluation*).  Kept here so the protocol value is one table rather than a
#: side effect of whichever runner's argparse default happens to apply;
#: ``--max-tokens`` overrides it.
DEFAULT_MAX_TOKENS = {
    "gaokao": 1024,
    "edumath": 2048,
    "mdk12": 1024,
    "tutorbench": 6144,
    "longtutor": 1024,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniedu judge", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", required=True, choices=["gaokao", "edumath", "mdk12", "tutorbench", "longtutor"]
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument(
        "--extra", nargs=argparse.REMAINDER, help="flags forwarded verbatim to the underlying judge"
    )
    return parser


def _common(args: argparse.Namespace) -> list[str]:
    result = [
        "--input",
        args.input,
        "--output",
        args.output,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
    ]
    if args.limit is not None:
        result += ["--limit", str(args.limit)]
    if args.allow_errors:
        result += ["--allow-errors"]
    return result + list(args.extra or [])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_tokens = args.max_tokens if args.max_tokens is not None else DEFAULT_MAX_TOKENS[args.task]
    if args.task == "longtutor":
        from . import judge_longtutor

        extra = ["--model", args.judge_model, *(_common(args)), "--max-tokens", str(max_tokens)]
        return judge_longtutor.main(extra)
    if args.task in ("mdk12", "tutorbench"):
        from . import judge_mdk12_tutorbench

        extra = [
            "--task",
            args.task,
            "--judge-model",
            args.judge_model,
            *_common(args),
            "--max-tokens",
            str(max_tokens),
        ]
        return _asyncio(judge_mdk12_tutorbench, extra)
    # GAOKAO subjective and EDUMATH run through the generic judge runner.
    from . import judge_gaokao_edumath

    common = _common(args) + ["--model", args.judge_model, "--max-tokens", str(max_tokens)]
    if args.task == "gaokao":
        common += [
            "--prompt-map",
            str(judge_prompt("gaokao_subject_prompts.json")),
            "--user-template",
            str(judge_prompt("gaokao_query.txt")),
            "--schema",
            "gaokao_subjective_official",
        ]
    else:
        common += [
            "--prompt-file",
            str(judge_prompt("edumath_official.txt")),
            "--user-template",
            str(judge_prompt("edumath_query.txt")),
            "--schema",
            "edumath_official",
        ]
    return _asyncio(judge_gaokao_edumath, common)


def _asyncio(module: object, argv: list[str]) -> int:
    import asyncio

    return int(asyncio.run(module.main(argv)))  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
