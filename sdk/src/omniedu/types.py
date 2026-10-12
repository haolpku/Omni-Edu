"""Small value types shared by the toolkit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class GenerationParams:
    """Decoding parameters.

    The defaults are the evaluation protocol of the paper: greedy decoding
    (``temperature=0``) and thinking disabled, because the models are trained
    no-think.
    """

    temperature: float = 0.0
    max_new_tokens: int = 2048
    enable_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merged(self, **overrides: Any) -> "GenerationParams":
        values = self.to_dict()
        values.update({key: value for key, value in overrides.items() if value is not None})
        return GenerationParams(**values)


@dataclass
class Message:
    """One chat message. ``content`` is a string or a list of content parts."""

    role: str
    content: Any

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    text: str
    model: str
    task: str | None = None
    system_prompt: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    finish_reason: str | None = None

    def __str__(self) -> str:
        return self.text

    def to_record(self) -> dict[str, Any]:
        """The reproducibility record written by :meth:`OmniEdu.batch`."""
        return {
            "response": self.text,
            "model": self.model,
            "task": self.task,
            "params": self.params,
            "latency_ms": round(self.latency_ms, 3),
            "finish_reason": self.finish_reason,
        }
