"""``omniedu eval``: run the paper's benchmarks against a served model.

The command is a thin dispatcher.  It resolves ``--benchmarks`` against the
registry, hands each benchmark the same namespace (endpoints, model, data root,
output directory, sampling), and writes one ``summary.json`` per benchmark.  A
benchmark module owns its own runner, flags and scorer; nothing here knows what
GPQA or TutorBench actually do.

    omniedu eval --model 4B --base-url http://127.0.0.1:8000/v1 --benchmarks all
    omniedu eval --benchmarks gpqa,ifeval --data-root evaluation_data --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from . import benchmarks as catalogue
from ..presets import is_alias, served_name
from .client import normalize_base_url
from .registry import Benchmark, groups, load_registry, slug
from .sampling import add_sampling_args


def _preparse(argv: Sequence[str] | None) -> str:
    """Read ``--benchmarks`` before the full parser exists.

    Benchmark-specific flags are only registered for the benchmarks actually
    selected, so the selection has to be known first.
    """
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--benchmarks", default="all")
    known, _ = probe.parse_known_args(argv)
    return known.benchmarks


def build_parser(selected: list[Benchmark]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniedu eval", description=__doc__.splitlines()[0])
    parser.add_argument("--benchmarks", default="all", help="names, groups, or all")
    parser.add_argument("--list", action="store_true", help="print the registry and exit")

    parser.add_argument("--model", help="served model name (vLLM --served-model-name)")
    parser.add_argument(
        "--base-url",
        dest="base_urls",
        action="append",
        help="OpenAI-compatible endpoint (repeatable)",
    )
    parser.add_argument("--api-key", default="EMPTY")

    parser.add_argument("--data-root", default="evaluation_data", help="root for benchmark inputs")
    parser.add_argument("--results-root", default="evaluation_results", help="root for run outputs")
    parser.add_argument(
        "--output-dir", default=None, help="override the per-model output directory"
    )
    parser.add_argument(
        "--output-tag", default=None, help="filesystem-safe tag (default: the model name)"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--benchmark-workers", type=int, default=4)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap rows per benchmark (smoke tests)"
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="this process's shard (EXAMS-V / MDK12 / TutorBench)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="total shards to split the workload into"
    )

    parser.add_argument(
        "--official-root",
        action="append",
        default=[],
        metavar="BENCH=PATH",
        help="upstream clone directory for one benchmark (repeatable)",
    )
    parser.add_argument(
        "--python", default=sys.executable, help="interpreter for upstream-driven benchmarks"
    )
    parser.add_argument(
        "--reward-python",
        default=sys.executable,
        help="interpreter for the MathTutorBench reward model",
    )
    parser.add_argument(
        "--omit-image",
        action="store_true",
        help="text-only ablation for EXAMS-V / MDK12 / TutorBench",
    )

    parser.add_argument("--num-samples", type=int, default=1000, help="generated items (EDUMATH)")
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve paths and print the plan; run nothing"
    )
    add_sampling_args(parser)
    for benchmark in selected:
        catalogue.module_for(benchmark).add_arguments(parser)
    return parser


def _parse_official(values: list[str]) -> dict[str, Path]:
    registry = load_registry()
    lookup = {slug(name): name for name in registry}
    resolved: dict[str, Path] = {}
    for raw in values:
        name, sep, path = str(raw).partition("=")
        if not sep or not path.strip():
            raise SystemExit(f"--official-root expects BENCH=PATH, got {raw!r}")
        key = slug(name)
        if key not in lookup:
            raise SystemExit(f"unknown benchmark in --official-root: {name!r}")
        resolved[lookup[key]] = Path(path).expanduser().resolve()
    return resolved


def _print_registry() -> int:
    registry = load_registry()
    for group, names in groups(registry).items():
        print(f"{group}")
        for name in names:
            benchmark = registry[name]
            judged = "judge" if benchmark.needs_judge else "local"
            print(f"  {name:<16} [{judged}] {benchmark.scoring}")
            source = benchmark.upstream.get("code")
            if source:
                print(f"  {'':<16}  {source}")
    return 0


def _run_one(benchmark: Benchmark, args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    module = catalogue.module_for(benchmark)
    namespace = argparse.Namespace(**vars(args))
    namespace.benchmark = benchmark.name
    namespace.output_dir = str(output_root / benchmark.name)
    namespace.official = getattr(args, "official", {})
    # Most runners are async; the ones that drive an upstream script from a
    # subprocess (K12-Bench, MathTutorBench) are plain functions.
    summary = module.run(namespace)
    if inspect.isawaitable(summary):
        summary = asyncio.run(summary)
    target = output_root / benchmark.name / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--list" in arguments:
        return _print_registry()
    registry = load_registry()
    selected = catalogue.select(_preparse(arguments), registry)
    parser = build_parser(selected)
    args = parser.parse_args(arguments)

    if args.list:
        return _print_registry()
    if not args.model or not args.base_urls:
        parser.error("--model and --base-url are required (unless --list)")

    args.base_urls = [normalize_base_url(url) for url in args.base_urls]
    official = _parse_official(args.official_root)
    setattr(args, "official", official)
    tag = slug(args.output_tag or args.model)
    # ``--model`` names the server.  A short alias means the user followed the
    # `omniedu serve` command, which serves it as ``omniedu-<size>``, so send the
    # name that command prints instead of the bare alias.
    if is_alias(args.model):
        args.model = served_name(args.model)
    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.results_root).expanduser().resolve() / "outputs" / tag
    )

    if args.dry_run:
        plan = {
            "model": args.model,
            "base_urls": args.base_urls,
            "data_root": str(Path(args.data_root).expanduser().resolve()),
            "output_root": str(output_root),
            "benchmarks": [benchmark.name for benchmark in selected],
            "official_overrides": {name: str(path) for name, path in official.items()},
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(args.benchmark_workers, len(selected)))
    ) as pool:
        futures = {
            pool.submit(_run_one, benchmark, args, output_root): benchmark for benchmark in selected
        }
        for future, benchmark in [(item, futures[item]) for item in futures]:
            try:
                results[benchmark.name] = future.result()
            except Exception as exc:  # keep other benchmarks running
                failures.append((benchmark.name, f"{type(exc).__name__}: {exc}"))

    print(json.dumps({"ok": sorted(results), "failed": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
