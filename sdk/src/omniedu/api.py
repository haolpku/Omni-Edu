"""The :class:`OmniEdu` entry point: load a checkpoint, chat, run a batch."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import media
from .backends import Backend, build_backend
from .io import (
    Request,
    append_jsonl,
    compact_jsonl,
    iter_jsonl,
    normalise_request,
)
from .presets import DEFAULT_GENERATION, is_alias, resolve_model, served_name
from .tasks import get_task
from .types import ChatResponse, GenerationParams


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _SyncLoop:
    """One private event loop per instance, for the synchronous entry points.

    ``asyncio.run`` closes its loop when it returns, which would leave the
    backend's async HTTP client bound to a dead loop: a following ``close()``
    (or a second call) fails with "Event loop is closed".  Keeping a single loop
    alive for the lifetime of the instance avoids that entirely.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None

    def run(self, coroutine: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # Never leave an un-awaited coroutine behind when we refuse to run it.
            coroutine.close()
            raise RuntimeError(
                "this call cannot run inside an active event loop; await the "
                "`a`-prefixed method (achat / abatch) instead"
            )
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coroutine)

    def close(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None


class OmniEdu:
    """A loaded Omni-Edu model (local or behind an OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str,
        backend: Backend,
        *,
        params: GenerationParams | None = None,
        concurrency: int = 8,
    ) -> None:
        self.model = model
        self.backend = backend
        self.params = params or DEFAULT_GENERATION
        self.concurrency = concurrency
        self._loop = _SyncLoop()

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_pretrained(
        cls,
        model: str = "4B",
        *,
        backend: str = "auto",
        base_url: str | None = None,
        api_key: str = "EMPTY",
        served_model: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        enable_thinking: bool = False,
        concurrency: int = 8,
        **backend_kwargs: Any,
    ) -> "OmniEdu":
        """Load ``"4B"`` / ``"9B"`` / ``"27B"``, a Hub id, or a local path.

        With ``base_url`` the toolkit talks to a running vLLM / OpenAI-compatible
        server; without it, it loads the weights locally (needs the
        ``omniedu[transformers]`` extra).

        A short alias (``"4B"`` / ``"9B"`` / ``"27B"``) against a ``base_url``
        sends the *served* name -- ``"omniedu-27b"`` -- because that is what
        ``omniedu serve`` passes to ``--served-model-name``.  ``served_model``
        overrides it; a full Hub id or a local path is sent unchanged.
        """
        resolved = resolve_model(model)
        request_model = served_model or (
            served_name(model) if base_url and is_alias(model) else resolved
        )
        instance_backend = build_backend(
            request_model, backend, base_url=base_url, api_key=api_key, **backend_kwargs
        )
        params = GenerationParams(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        return cls(resolved, instance_backend, params=params, concurrency=concurrency)

    # ------------------------------------------------------------------ messages

    def _compose(
        self,
        *,
        text: str | None,
        images: Any,
        messages: Sequence[dict[str, Any]] | None,
        task: str | None,
        system: str | None,
    ) -> tuple[list[dict[str, Any]], str | None, str]:
        """Return ``(messages, system_prompt, task_id)`` ready for the backend."""
        instruction = system
        task_id = task or ""
        if task and not instruction:
            instruction = get_task(task).instruction

        urls = media.normalise_images(images)
        if self.backend.name == "transformers":
            # The local processor wants PIL images, not URL references.
            def content(value: str) -> Any:
                return media.to_transformers_content(value, urls)

        else:

            def content(value: str) -> Any:
                return media.build_user_content(value, urls)

        if messages:
            built = []
            for message in messages:
                message = dict(message)
                value = message.get("content")
                if isinstance(value, str) and (urls or media.IMAGE_MARKER in value):
                    message["content"] = content(value)
                built.append(message)
        else:
            built = [{"role": "user", "content": content(text or "")}]

        if instruction:
            built = [{"role": "system", "content": instruction}] + built
        return built, instruction, task_id

    # ------------------------------------------------------------------ single

    async def achat(
        self,
        *,
        text: str | None = None,
        image: Any = None,
        images: Any = None,
        messages: Sequence[dict[str, Any]] | None = None,
        task: str | None = None,
        system: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        enable_thinking: bool | None = None,
    ) -> ChatResponse:
        """Generate one reply."""
        params = self.params.merged(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
        # `images` wins when given; both accept a single image or an iterable.
        supplied = images if images is not None else image
        built, instruction, task_id = self._compose(
            text=text, images=supplied, messages=messages, task=task, system=system
        )
        started = time.perf_counter()
        generation = await self.backend.generate(built, params)
        return ChatResponse(
            text=generation.text,
            model=self.model,
            task=task_id or None,
            system_prompt=instruction,
            params=params.to_dict(),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            finish_reason=generation.finish_reason,
        )

    def chat(self, **kwargs: Any) -> ChatResponse:
        """Synchronous :meth:`achat`."""
        return self._loop.run(self.achat(**kwargs))

    # ------------------------------------------------------------------ batch

    async def abatch(
        self,
        source: str | Path | Iterable[dict[str, Any]],
        out: str | Path | None = None,
        *,
        task: str | None = None,
        system: str | None = None,
        concurrency: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        limit: int | None = None,
        resume: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Run a JSONL (or in-memory) batch and append results to ``out``.

        Only successful rows count as done, so a re-run retries failures; the
        output file is compacted first so a retry never leaves two lines for the
        same sample.
        """
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            rows = list(iter_jsonl(source_path))
            base_dir = source_path.parent
        else:
            rows = list(source)
            base_dir = None

        requests: list[Request] = []
        for index, row in enumerate(rows):
            requests.append(normalise_request(row, index=index, task=task, system=system))
        if limit is not None:
            requests = requests[:limit]

        done: set[str] = set()
        previous = 0
        if out is not None and resume:
            out_path = Path(out)
            if out_path.exists():
                for record in iter_jsonl(out_path):
                    previous += 1
                    if not record.get("error"):
                        done.add(str(record.get("id")))
                compact_jsonl(out_path, key="id", keep=done)

        params = self.params.merged(max_new_tokens=max_new_tokens, temperature=temperature)
        workers = concurrency or self.concurrency
        lock = asyncio.Lock()
        stats = {"ok": 0, "error": 0, "skipped": 0}
        finished = 0
        total = len(requests)

        async def handle(request: Request) -> None:
            nonlocal finished
            if request.id in done:
                async with lock:
                    stats["skipped"] += 1
                    finished += 1
                return
            record: dict[str, Any] = {"id": request.id}
            try:
                response = await self.achat(
                    messages=request.messages,
                    images=request.images,
                    task=request.task,
                    system=request.system,
                    max_new_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    enable_thinking=params.enable_thinking,
                )
                record.update(response.to_record())
                record["system_prompt_sha256"] = (
                    _sha256(response.system_prompt) if response.system_prompt else None
                )
                record["error"] = None
                key = "ok"
            except Exception as exc:  # noqa: BLE001 - recorded per row, not fatal
                record.update({"response": "", "error": f"{type(exc).__name__}: {exc}"})
                key = "error"
            async with lock:
                if out is not None:
                    append_jsonl(out, record)
                stats[key] += 1
                finished += 1
                if progress is not None:
                    progress(finished, total)
                elif finished % 50 == 0:
                    print(
                        f"[omniedu] {finished}/{total} ok={stats['ok']} error={stats['error']}",
                        flush=True,
                    )

        semaphore = asyncio.Semaphore(workers)

        async def guarded(request: Request) -> None:
            async with semaphore:
                await handle(request)

        await asyncio.gather(*(guarded(request) for request in requests))
        return {
            "model": self.model,
            "total": total,
            "previous_records": previous,
            "skipped_already_done": stats["skipped"],
            "ok": stats["ok"],
            "error": stats["error"],
            "output": str(out) if out is not None else None,
        }

    def batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Synchronous :meth:`abatch`."""
        return self._loop.run(self.abatch(*args, **kwargs))

    # ------------------------------------------------------------------ teardown

    async def aclose(self) -> None:
        try:
            await self.backend.aclose()
        except RuntimeError:
            # The client can only be on another loop if the caller mixed the
            # async API with the sync one; there is nothing left to release.
            pass

    def close(self) -> None:
        """Release the backend. Safe to call more than once."""
        try:
            self._loop.run(self.aclose())
        finally:
            self._loop.close()

    def __enter__(self) -> "OmniEdu":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"OmniEdu(model={self.model!r}, backend={self.backend.name!r})"
