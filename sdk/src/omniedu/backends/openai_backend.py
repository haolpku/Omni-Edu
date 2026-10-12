"""Backend for any OpenAI-compatible chat endpoint (vLLM, SGLang, hosted APIs)."""

from __future__ import annotations

from typing import Any

from ..types import GenerationParams
from .base import Generation


def api_base(url: str) -> str:
    """Accept both ``http://host:port`` and ``http://host:port/v1``.

    The OpenAI client appends ``/chat/completions`` to whatever base URL it gets,
    so blindly appending ``/v1`` turns the documented ``https://api.openai.com/v1``
    form into ``/v1/v1/chat/completions``.
    """
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class OpenAIBackend:
    name = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 600.0,
        max_retries: int = 2,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.base_url = api_base(base_url)
        self.extra_body = dict(extra_body or {})
        self._client = AsyncOpenAI(
            base_url=self.base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    async def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> Generation:
        body = dict(self.extra_body)
        template_kwargs = dict(body.get("chat_template_kwargs") or {})
        template_kwargs.setdefault("enable_thinking", params.enable_thinking)
        body["chat_template_kwargs"] = template_kwargs
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": params.max_new_tokens,
            "extra_body": body,
        }
        # Greedy decoding: sending temperature=0 alongside no sampling keeps the
        # request identical to the evaluation protocol.
        request["temperature"] = params.temperature
        completion = await self._client.chat.completions.create(**request)
        choice = completion.choices[0]
        return Generation(
            text=choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None),
            raw={"base_url": self.base_url, "model": self.model},
        )

    async def aclose(self) -> None:
        await self._client.close()
