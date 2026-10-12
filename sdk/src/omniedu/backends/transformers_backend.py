"""Local inference with ``transformers``.

This path loads the checkpoint into the current process and is meant for single
machine use / verification. For throughput, serve the model with vLLM and use
the OpenAI-compatible backend instead.

Status: written against the usage published on the model cards
(``AutoProcessor`` + ``AutoModelForImageTextToText``). It has not been exercised
on a GPU in this repository, so treat it as the less-travelled path.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..types import GenerationParams
from .base import Generation


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        model: str,
        *,
        dtype: str | Any = "bfloat16",
        device_map: str | dict[str, Any] = "auto",
        trust_remote_code: bool = True,
        **load_kwargs: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self.model_id = model
        # One local model cannot serve concurrent generate() calls.
        self._lock = asyncio.Semaphore(1)

        self._processor = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        kwargs: dict[str, Any] = dict(load_kwargs)
        kwargs["device_map"] = device_map
        kwargs["trust_remote_code"] = trust_remote_code
        if dtype is not None:
            kwargs["dtype"] = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(model, **kwargs)
        except TypeError:
            # transformers < 5 spells this ``torch_dtype``.
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = AutoModelForImageTextToText.from_pretrained(model, **kwargs)
        self._model.eval()

    def _apply_template(self, messages: list[dict[str, Any]], enable_thinking: bool) -> Any:
        common = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        try:
            return self._processor.apply_chat_template(
                messages, enable_thinking=enable_thinking, **common
            )
        except TypeError:
            # Templates that do not accept the flag simply omit it.
            return self._processor.apply_chat_template(messages, **common)

    def _generate_sync(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> Generation:
        inputs = self._apply_template(messages, params.enable_thinking).to(self._model.device)
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": params.temperature > 0,
        }
        if params.temperature > 0:
            generate_kwargs["temperature"] = params.temperature
        with self._torch.inference_mode():
            output = self._model.generate(**inputs, **generate_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        text = self._processor.decode(output[0][prompt_length:], skip_special_tokens=True)
        return Generation(text=text.strip(), finish_reason=None, raw={"model": self.model_id})

    async def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams
    ) -> Generation:
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, messages, params)

    async def aclose(self) -> None:
        return None
