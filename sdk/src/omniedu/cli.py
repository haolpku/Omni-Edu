"""Command line entry point: ``omniedu tasks|chat|batch|serve|eval``."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from typing import Any, Sequence

from . import __version__
from .api import OmniEdu
from .presets import known_presets, resolve_model, served_name
from .tasks import get_task, list_tasks


def _add_endpoint(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="4B", help="4B / 9B / 27B, a Hub id, or a local path")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1"
    )
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--served-model", help="name sent in requests (vLLM --served-model-name)")
    parser.add_argument("--backend", default="auto", choices=["auto", "openai", "transformers"])


def _add_generation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", help="teaching task id, see `omniedu tasks`")
    parser.add_argument("--system", help="explicit system prompt (overrides --task)")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)


def _cmd_tasks(args: argparse.Namespace) -> int:
    tasks = list_tasks()
    if args.show:
        task = get_task(args.show)
        print(f"# {task.id}  [{task.category}]\n\n{task.instruction}")
        return 0
    width = max(len(name) for name in tasks)
    for name in sorted(tasks, key=lambda item: (tasks[item].category, item)):
        task = tasks[name]
        if args.category and task.category != args.category:
            continue
        # Instructions can be multi-line (the LongTutor one is a whole spec);
        # collapse so the listing stays one row per task.
        summary = " ".join(task.instruction.split())[:70]
        print(f"{name:<{width}}  {task.category:<24} {summary}")
    return 0


def _build_model(args: argparse.Namespace) -> OmniEdu:
    return OmniEdu.from_pretrained(
        args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        served_model=args.served_model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        concurrency=getattr(args, "concurrency", 8),
    )


def _cmd_chat(args: argparse.Namespace) -> int:
    model = _build_model(args)
    try:
        response = model.chat(
            text=args.text,
            images=args.image or None,
            task=args.task,
            system=args.system,
        )
    finally:
        model.close()
    print(response.text)
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    model = _build_model(args)
    try:
        summary = model.batch(
            args.input,
            out=args.output,
            task=args.task,
            system=args.system,
            concurrency=args.concurrency,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            limit=args.limit,
        )
    finally:
        model.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if summary["error"] else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    repo = resolve_model(args.model)
    served = args.served_model or served_name(args.model)
    command = [
        "vllm",
        "serve",
        repo,
        "--served-model-name",
        served,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt",
        json.dumps({"image": args.max_images}),
        "--trust-remote-code",
    ]
    # shlex.join, not " ".join: the `--limit-mm-per-prompt` value contains a
    # space, so a plain join prints a command the shell would split in two.
    print(shlex.join(command))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniedu", description="Omni-Edu model toolkit")
    parser.add_argument("--version", action="version", version=f"omniedu {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tasks = subparsers.add_parser("tasks", help="list the shipped teaching instructions")
    tasks.add_argument("--category", help="only this category")
    tasks.add_argument("--show", help="print the full instruction for one task id")
    tasks.set_defaults(func=_cmd_tasks)

    chat = subparsers.add_parser("chat", help="one text/image turn")
    _add_endpoint(chat)
    _add_generation(chat)
    chat.add_argument("--text", required=True)
    chat.add_argument("--image", action="append", help="repeatable")
    chat.set_defaults(func=_cmd_chat)

    batch = subparsers.add_parser("batch", help="JSONL in, JSONL out")
    _add_endpoint(batch)
    _add_generation(batch)
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--concurrency", type=int, default=8)
    batch.add_argument("--limit", type=int)
    batch.set_defaults(func=_cmd_batch)

    serve = subparsers.add_parser("serve", help="print the matching vLLM command")
    serve.add_argument("--model", default="27B")
    serve.add_argument("--served-model")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-model-len", type=int, default=32768)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    serve.add_argument("--max-images", type=int, default=12)
    serve.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # ``eval`` owns a large, self-contained parser (one flag set per benchmark),
    # so it is dispatched before the core parser is built rather than inlined.
    if arguments[:1] == ["eval"]:
        from .eval.cli import main as eval_main

        return eval_main(arguments[1:])
    if arguments[:1] == ["judge"]:
        from .eval.judge.cli import main as judge_main

        return judge_main(arguments[1:])
    if arguments[:1] == ["report"]:
        from .eval.report import main as report_main

        return report_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive convenience
        return 130
    except Exception as exc:  # noqa: BLE001 - surface a one-line error, not a traceback
        print(f"omniedu: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def presets_help() -> dict[str, str]:
    """Exposed for tests and docs."""
    return known_presets()
