"""The pedagogical task instructions shipped with the Omni-Edu mixture.

These are the instructions the models were fine-tuned with, so using the
matching one at inference keeps you in the distribution the model was trained
on. Note that the paper's evaluation protocol deliberately does **not** add
them when scoring a benchmark.

``tasks.json`` is generated from the single source of truth in the research
repository (``data_preparation/stage6_.../assign_system_prompts.py``) by
``sdk/tools/sync_tasks.py``. The released package carries its own copy so it can
be installed on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    category: str
    source: str | None = None

    def __str__(self) -> str:
        return self.instruction


_REGISTRY: dict[str, Task] | None = None


def _load() -> dict[str, Task]:
    global _REGISTRY
    if _REGISTRY is None:
        payload = resources.files(__package__).joinpath("tasks.json").read_text(encoding="utf-8")
        data = json.loads(payload)
        _REGISTRY = {
            name: Task(
                id=name,
                instruction=spec["instruction"],
                category=spec.get("category", "other"),
                source=spec.get("source"),
            )
            for name, spec in data["tasks"].items()
        }
    return _REGISTRY


def list_tasks() -> dict[str, Task]:
    """Every registered task, keyed by id."""
    return dict(_load())


def get_task(name: str) -> Task:
    try:
        return _load()[name]
    except KeyError:
        known = ", ".join(sorted(_load()))
        raise KeyError(f"unknown task {name!r}; available: {known}") from None


def register_task(name: str, instruction: str, *, category: str = "custom") -> Task:
    """Add or replace a task at runtime.

    This is the extension point for users who need their own teaching
    instruction; it never touches the installed package.
    """
    if not name or not instruction:
        raise ValueError("name and instruction must be non-empty")
    task = Task(id=name, instruction=instruction, category=category)
    _load()[name] = task
    return task


def tasks_version() -> str:
    payload = resources.files(__package__).joinpath("tasks.json").read_text(encoding="utf-8")
    return json.loads(payload).get("version", "unknown")
