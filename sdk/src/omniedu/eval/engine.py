"""The generation engine every benchmark runner shares.

The runners all do the same three things: rotate over one or more served
replicas, retry a request that the transport dropped, and write one JSON line
per sample so an interrupted run can resume.  This module is that loop.

Requests are sent with ``httpx`` and an explicit JSON body rather than through
an SDK wrapper, so the bytes that reach the model are exactly the dict the
runner built - a refactor can be checked against a recorded baseline by diffing
those bodies.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import httpx

from .client import EndpointPool, normalize_base_url
from .io import append_jsonl

#: A request that could not be delivered after every retry stores this marker in
#: the response field.  The resume rule treats such a row as "not done", so the
#: next run retries it instead of freezing the failure into a score.
ERROR_PREFIX = "__ERROR__: "

#: The educational benchmarks are all run with the served model's thinking mode
#: disabled, as the paper's protocol specifies.  The general-capability runners
#: (GPQA / IFEval / MMMU-Pro) leave it to ``--chat-template-kwargs`` instead.
THINKING_OFF: dict[str, bool] = {"enable_thinking": False}


def endpoints(values: Iterable[str]) -> list[str]:
    """Normalise ``--base-url`` / ``--endpoints`` values to ``.../v1`` bases."""
    return [normalize_base_url(value) for value in values]


async def chat(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    *,
    retries: int = 4,
    backoff: float = 2.0,
) -> tuple[str, str | None]:
    """POST one chat completion and return ``(content, finish_reason)``.

    On a persistent transport error the content is an ``__ERROR__`` marker and
    the finish reason is ``None``; callers treat that as a missing answer.
    """
    attempts = max(1, retries)
    error: Any = None
    for attempt in range(attempts):
        try:
            response = await client.post(f"{url}/chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            return choice["message"]["content"] or "", choice.get("finish_reason")
        except Exception as exc:  # transport, HTTP status, or an unexpected body
            error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(min(backoff * 2**attempt, 16.0))
    return f"{ERROR_PREFIX}{error!r}", None


def marker_failed(value: Any) -> bool:
    """True when a stored response field is an ``__ERROR__`` marker."""
    return str(value or "").startswith(ERROR_PREFIX)


async def fan_out(
    items: Sequence[Any],
    worker: Callable[[int, Any], Awaitable[None]],
    *,
    concurrency: int,
    chunk: int | None = None,
) -> None:
    """Run ``worker(index, item)`` over ``items`` with bounded concurrency.

    The whole list is scheduled at once and a semaphore caps in-flight work.  A
    caller whose items hold large payloads (parquet-row images) can pass
    ``chunk`` to await in batches instead, so only a window is ever in memory.
    """
    gate = asyncio.Semaphore(concurrency)

    async def guarded(index: int, item: Any) -> None:
        async with gate:
            await worker(index, item)

    if chunk is None:
        await asyncio.gather(*(guarded(index, item) for index, item in enumerate(items)))
        return
    for start in range(0, len(items), chunk):
        window = items[start : start + chunk]
        await asyncio.gather(*(guarded(start + offset, item) for offset, item in enumerate(window)))


class Sink:
    """One JSONL output file, appended under a lock, flushed every line."""

    def __init__(self, path: str | Path, **dumps: Any) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dumps = dumps
        self._lock = asyncio.Lock()
        self.count = 0

    async def write(self, record: dict[str, Any]) -> None:
        async with self._lock:
            append_jsonl(self.path, record, **self.dumps)
            self.count += 1


def clients(urls: Sequence[str], *, timeout: float) -> httpx.AsyncClient:
    """One client for all endpoints (the pool rotates the URLs, not the client)."""
    return httpx.AsyncClient(timeout=timeout)


def picker(urls: Sequence[str]) -> EndpointPool:
    return EndpointPool(list(urls))
