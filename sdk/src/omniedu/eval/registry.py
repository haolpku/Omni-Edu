"""The benchmark registry: what ``omniedu eval`` can run, and how.

``benchmarks.json`` records, per benchmark, which upstream project and dataset it
comes from, the module that drives it, whether an LLM judge scores any part of
it, and which paper table it feeds.  Shipping it inside the package means the
CLI and the documentation cannot disagree about what exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Benchmark:
    name: str
    group: str
    module: str
    data: tuple[str, ...] = ()
    upstream: dict[str, Any] = field(default_factory=dict)
    judge: dict[str, Any] | None = None
    metrics: tuple[str, ...] = ()
    scoring: str = ""
    paper_table: int | None = None
    subtasks: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()

    @property
    def needs_judge(self) -> bool:
        return bool(self.judge)

    @property
    def judged_by(self) -> str | None:
        return (self.judge or {}).get("runner")


def _from_entry(entry: dict[str, Any]) -> Benchmark:
    return Benchmark(
        name=entry["name"],
        group=entry["group"],
        module=entry["module"],
        data=tuple(entry.get("data") or ()),
        upstream=entry.get("upstream") or {},
        judge=entry.get("judge"),
        metrics=tuple(entry.get("metrics") or ()),
        scoring=entry.get("scoring") or "",
        paper_table=entry.get("paper_table"),
        subtasks=tuple(entry.get("subtasks") or ()),
        tasks=tuple(entry.get("tasks") or ()),
    )


def load_registry(path: str | Path | None = None) -> dict[str, Benchmark]:
    """Return the registry keyed by benchmark name.

    With no argument the packaged ``benchmarks.json`` is used, so an installed
    copy is self-contained.  Passing a path points the code at a different
    registry file.
    """
    if path is None:
        payload = (
            resources.files(__package__).joinpath("benchmarks.json").read_text(encoding="utf-8")
        )
    else:
        payload = Path(path).read_text(encoding="utf-8")
    data = json.loads(payload)
    return {entry["name"]: _from_entry(entry) for entry in data["entries"]}


def groups(registry: dict[str, Benchmark]) -> dict[str, list[str]]:
    """Benchmark names grouped by capability, in registry order."""
    grouped: dict[str, list[str]] = {}
    for benchmark in registry.values():
        grouped.setdefault(benchmark.group, []).append(benchmark.name)
    return grouped


def slug(value: str) -> str:
    """Lowercase, punctuation-free key used for tolerant name matching."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def resolve(name: str, registry: dict[str, Benchmark] | None = None) -> Benchmark:
    """Look a benchmark up by name, ignoring case and punctuation.

    ``gpqa``, ``GPQA`` and ``mmmu-pro`` all resolve to their registry entry, so
    the CLI can accept the spellings people actually type.
    """
    registry = registry or load_registry()
    key = slug(name)
    for benchmark in registry.values():
        if slug(benchmark.name) == key:
            return benchmark
    raise KeyError(name)
