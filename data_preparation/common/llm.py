#!/usr/bin/env python3
"""Shared LLM client: OpenAI-compatible endpoint pool, concurrency, retries, resume.

Auditing, rewriting, re-auditing and quality scoring all use this client; no stage implements its own
HTTP calls. Endpoints and model vary per deployment, so they are resolved as command line →
environment variable → default, with no machine-specific values in the code:

    python audit.py --dataset FEAT --endpoints "<vLLM endpoints>" --concurrency <value>
    AUDIT_ENDPOINTS="<endpoints>" AUDIT_MODEL="<model>" python audit.py --dataset FEAT
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from openai import AsyncOpenAI

DEFAULT_ENDPOINTS = "http://127.0.0.1:8400/v1"
DEFAULT_MODEL = "Qwen3.5-122B"
#: default concurrency per endpoint; override per machine
CONCURRENCY_PER_ENDPOINT = 32
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


def endpoints(value: str | None = None) -> list[str]:
    """Command line first, then environment, then the default. Space- or comma-separated."""
    raw = value or os.environ.get("AUDIT_ENDPOINTS") or DEFAULT_ENDPOINTS
    return [item for chunk in raw.split() for item in chunk.split(",") if item]


def model_name(value: str | None = None) -> str:
    return value or os.environ.get("AUDIT_MODEL") or DEFAULT_MODEL


def default_concurrency(urls: list[str] | None = None) -> int:
    """Default concurrency scales with the number of endpoints."""
    return CONCURRENCY_PER_ENDPOINT * max(1, len(urls or endpoints()))


def add_common_args(parser: "argparse.ArgumentParser") -> None:
    """Connection options shared by audit, rewrite and generate; all overridable."""
    parser.add_argument(
        "--endpoints",
        default=None,
        help=f"vLLM endpoints, space- or comma-separated; from AUDIT_ENDPOINTS, then {DEFAULT_ENDPOINTS}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"served model name; from AUDIT_MODEL, then {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=f"concurrent requests; default {CONCURRENCY_PER_ENDPOINT} per endpoint",
    )
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--attempts", type=int, default=3)


def strip_think(text: str) -> str:
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def parse_json(text: str) -> dict[str, Any] | None:
    match = JSON_RE.search(text or "")
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class Client:
    """Round-robins over replicas and returns parsed JSON only."""

    def __init__(
        self,
        urls: list[str] | None = None,
        model: str | None = None,
        *,
        timeout: float = 300.0,
        attempts: int = 3,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ):
        self.clients = [
            AsyncOpenAI(base_url=url, api_key="EMPTY", timeout=timeout, max_retries=0)
            for url in (urls or endpoints())
        ]
        self.model = model_name(model)
        self.attempts = attempts
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cycle = itertools.cycle(self.clients)

    async def json_call(
        self,
        system: str,
        user: str,
        *,
        attempts: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        error = ""
        for attempt in range(attempts or self.attempts):
            client = next(self._cycle)
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                obj = parse_json(strip_think(response.choices[0].message.content))
                if obj is not None:
                    return obj, ""
                error = "unparsable_response"
            except Exception as exc:  # noqa: BLE001 - retry instead of aborting the batch
                error = f"{type(exc).__name__}: {exc}"[:200]
                await asyncio.sleep(2 * (attempt + 1))
        return None, error


def from_args(args: Any) -> Client:
    """Build a client from parsed arguments (endpoints, model, timeout, retries, output length)."""
    return Client(
        endpoints(getattr(args, "endpoints", None)),
        getattr(args, "model", None),
        timeout=getattr(args, "timeout", 300.0),
        attempts=getattr(args, "attempts", 3),
        max_tokens=getattr(args, "max_tokens", 8192),
    )


def load_done(path: Path, key: str, ok: Callable[[dict], bool]) -> set[str]:
    """Resume support: read back the keys that already succeeded."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ok(row):
                done.add(str(row[key]))
    return done


async def run(
    records: list[dict[str, Any]],
    out_path: Path,
    prompt_of: Callable[[dict[str, Any]], tuple[str, str]],
    result_of: Callable[[dict[str, Any], dict[str, Any] | None, str], dict[str, Any]],
    *,
    chat: Client | None = None,
    key: str = "uid",
    concurrency: int | None = None,
    log_every: int = 500,
) -> dict[str, int]:
    """Run a batch concurrently, appending row by row so runs can resume."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client = chat or Client()
    workers = concurrency or default_concurrency()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for record in records:
        queue.put_nowait(record)

    stats: dict[str, int] = {}
    done = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal done
        while True:
            try:
                record = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            system, user = prompt_of(record)
            obj, error = await client.json_call(system, user)
            result = result_of(record, obj, error)
            async with lock:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                label = str(result.get("verdict") or result.get("status") or "error")
                stats[label] = stats.get(label, 0) + 1
                done += 1
                if log_every and done % log_every == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] done={done} stats={stats}", flush=True)
            queue.task_done()

    with out_path.open("a", encoding="utf-8") as handle:
        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(workers)])
    return stats
