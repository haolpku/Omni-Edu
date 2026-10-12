"""The backend contract: one coroutine that turns messages into text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import GenerationParams


@dataclass
class Generation:
    text: str
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Backend(Protocol):
    #: Short identifier used in error messages and reproducibility records.
    name: str

    async def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> Generation: ...

    async def aclose(self) -> None: ...
