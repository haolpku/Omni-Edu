"""Endpoint rotation, base-URL normalisation and client-side rate limiting.

Every runner used to build its own client and round-robin over replicas by hand,
and both judge runners had written their own token-bucket limiter.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Iterator


def normalize_base_url(url: str) -> str:
    """Accept both ``http://host:port`` and ``http://host:port/v1``.

    The OpenAI client appends ``/chat/completions`` to whatever it is given, so
    appending ``/v1`` unconditionally turns the documented ``.../v1`` form into
    ``/v1/v1/chat/completions``.
    """
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class EndpointPool:
    """Round-robin over one or more served replicas."""

    def __init__(self, endpoints: list[str] | tuple[str, ...]) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint is required")
        self.endpoints = list(endpoints)
        self._cycle: Iterator[str] = itertools.cycle(self.endpoints)

    def next(self) -> str:
        return next(self._cycle)

    def __len__(self) -> int:
        return len(self.endpoints)


class RateLimiter:
    """Client-side cap on requests per minute (``0`` disables it)."""

    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / per_minute if per_minute and per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval
