"""Sampling knobs, ``strip_think`` and the generation block for summaries.

``gpqa``, ``ifeval`` and ``mmmu_pro`` carried ~60 identical lines of this, held
together by module-level globals that ``main()`` mutated.  One dataclass and one
argparse helper replace all of it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any


def strip_think(text: str, *, enabled: bool = True) -> str:
    """Return the text after the final ``</think>`` when thinking mode is on.

    The runners only use this when the model is served with thinking enabled but
    the scorer wants the answer segment; with ``enabled`` false the text is
    returned unchanged, matching the greedy path where nothing is stripped.
    """
    if not isinstance(text, str) or not enabled:
        return text
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    return ""


@dataclass
class Sampling:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    seed: int | None = None
    max_tokens: int | None = None
    chat_template_kwargs: dict[str, Any] | None = None
    strip_thinking: bool = False

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Only touch sampling knobs when actually sampling.

        Greedy payloads stay byte-identical to what the runners sent before.
        """
        if self.temperature and self.temperature > 0:
            payload["top_p"] = self.top_p
            if self.top_k and self.top_k > 0:
                payload["top_k"] = self.top_k
            if self.presence_penalty:
                payload["presence_penalty"] = self.presence_penalty
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def describe(self) -> str:
        if self.temperature and self.temperature > 0:
            text = (
                f"sampled T={self.temperature} top_p={self.top_p} "
                f"top_k={self.top_k} pp={self.presence_penalty}"
            )
        else:
            text = "greedy"
        if self.seed is not None:
            text += f" seed={self.seed}"
        mode = "thinking-default"
        if self.chat_template_kwargs and self.chat_template_kwargs.get("enable_thinking") is False:
            mode = "thinking-off"
        return f"{text}, max_tokens={self.max_tokens}, mode={mode}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": self.chat_template_kwargs,
            "strip_thinking": self.strip_thinking,
        }

    def attach(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Record the parameters actually used, as the runners' summaries do."""
        summary["generation_params"] = self.to_dict()
        summary["protocol"] = summary["protocol"] + " || ACTUAL generation: " + self.describe()
        return summary


def add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """The sampling flags every runner accepts.

    ``--max-tokens`` defaults to ``None`` so each benchmark can apply its own
    upstream budget (GPQA 1000, IFEval 1280, MMMU-Pro 4096, ...) instead of one
    shared guess.
    """
    parser.add_argument("--chat-template-kwargs", default=None)
    parser.add_argument("--strip-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)


def sampling_from_args(
    args: argparse.Namespace, *, default_max_tokens: int | None = None
) -> Sampling:
    raw = getattr(args, "chat_template_kwargs", None)
    max_tokens = getattr(args, "max_tokens", None)
    return Sampling(
        temperature=getattr(args, "temperature", 0.0),
        top_p=getattr(args, "top_p", 1.0),
        top_k=getattr(args, "top_k", -1),
        presence_penalty=getattr(args, "presence_penalty", 0.0),
        seed=getattr(args, "seed", None),
        max_tokens=default_max_tokens if max_tokens is None else max_tokens,
        chat_template_kwargs=json.loads(raw) if raw else None,
        strip_thinking=bool(getattr(args, "strip_thinking", False)),
    )
