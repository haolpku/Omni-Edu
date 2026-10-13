"""One module per benchmark, each exposing ``add_arguments`` and ``run``.

A benchmark module owns its runner *and* its scorer, because the two share a
protocol (the prompt, the fields sent to the model, the parser).  The registry in
``benchmarks.json`` names the module; this package only knows how to load it.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from types import ModuleType

from ..registry import Benchmark, groups, load_registry, slug


def module_for(benchmark: Benchmark) -> ModuleType:
    return importlib.import_module(benchmark.module)


def select(
    names: str | Iterable[str] | None, registry: dict[str, Benchmark] | None = None
) -> list[Benchmark]:
    """Resolve ``--benchmarks`` to registry entries.

    Accepts ``all``, a comma-separated list of benchmark names, and capability
    group names (``curriculum_grounding``, ``problem_solving``,
    ``pedagogical_tutoring``, ``general_capability``).
    """
    registry = registry or load_registry()
    if names is None:
        return list(registry.values())
    if isinstance(names, str):
        items = [item for item in names.replace(",", " ").split() if item]
    else:
        items = [str(item) for item in names]
    if not items or any(item.lower() == "all" for item in items):
        return list(registry.values())

    by_group = groups(registry)
    group_index = {slug(name): name for name in by_group}
    lookup = {slug(benchmark.name): benchmark for benchmark in registry.values()}
    chosen: list[Benchmark] = []
    seen: set[str] = set()
    for item in items:
        key = slug(item)
        if key in lookup:
            bench = lookup[key]
            if bench.name not in seen:
                chosen.append(bench)
                seen.add(bench.name)
            continue
        if key in group_index:
            for name in by_group[group_index[key]]:
                if name not in seen:
                    chosen.append(registry[name])
                    seen.add(name)
            continue
        available = ", ".join(benchmark.name for benchmark in registry.values())
        raise SystemExit(f"unknown benchmark or group {item!r}; available: {available}")
    return chosen
