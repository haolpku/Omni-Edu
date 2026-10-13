import asyncio
import json

import pytest

from omniedu import OmniEdu, get_task
from omniedu.backends.base import Generation
from omniedu.io import load_jsonl


class StubBackend:
    """Records what it was asked and echoes a marker back."""

    name = "stub"

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        self.fail_on = fail_on
        self.calls: list[list[dict]] = []

    async def generate(self, messages, params):
        self.calls.append(messages)
        payload = json.dumps(messages, ensure_ascii=False)
        for needle in self.fail_on:
            if needle in payload:
                raise RuntimeError("boom")
        return Generation(text="echo", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_chat_prepends_the_task_instruction() -> None:
    backend = StubBackend()
    model = OmniEdu("stub-model", backend)
    response = model.chat(text="hi", task="solve_reasoned")

    sent = backend.calls[0]
    assert sent[0] == {"role": "system", "content": get_task("solve_reasoned").instruction}
    assert sent[1] == {"role": "user", "content": "hi"}
    assert response.text == "echo"
    assert response.task == "solve_reasoned"
    assert response.params["temperature"] == 0.0
    assert response.params["enable_thinking"] is False


def test_explicit_system_overrides_the_task() -> None:
    backend = StubBackend()
    model = OmniEdu("stub-model", backend)
    model.chat(text="hi", task="solve_reasoned", system="my own")
    assert backend.calls[0][0] == {"role": "system", "content": "my own"}


def test_generation_overrides_are_per_call() -> None:
    backend = StubBackend()
    model = OmniEdu("stub-model", backend)
    response = model.chat(text="hi", max_new_tokens=99, temperature=0.7)
    assert response.params["max_new_tokens"] == 99
    assert response.params["temperature"] == 0.7


def test_batch_writes_reproducible_records(tmp_path) -> None:
    rows = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
    out = tmp_path / "preds.jsonl"
    model = OmniEdu("stub-model", StubBackend(), concurrency=2)

    summary = model.batch(rows, out=out, task="solve_answer_only")
    assert summary["ok"] == 2 and summary["error"] == 0

    records = load_jsonl(out)
    assert [record["id"] for record in records] == ["a", "b"]
    for record in records:
        assert record["response"] == "echo"
        assert record["task"] == "solve_answer_only"
        assert record["system_prompt_sha256"]
        assert record["params"]["temperature"] == 0.0
        assert record["error"] is None


def test_batch_retries_failures_without_duplicating_rows(tmp_path) -> None:
    rows = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
    out = tmp_path / "preds.jsonl"

    failing = OmniEdu("stub-model", StubBackend(fail_on=("two",)), concurrency=2)
    first = failing.batch(rows, out=out)
    assert first["ok"] == 1 and first["error"] == 1
    assert {record["id"] for record in load_jsonl(out)} == {"a", "b"}

    # The failed row is not treated as done, and the retry must not leave two
    # lines for the sample that already succeeded.
    working = OmniEdu("stub-model", StubBackend(), concurrency=2)
    second = working.batch(rows, out=out)
    assert second["skipped_already_done"] == 1
    assert second["ok"] == 1

    records = load_jsonl(out)
    assert len(records) == 2
    assert all(record["error"] is None for record in records)


def test_batch_reports_row_errors_without_aborting(tmp_path) -> None:
    rows = [{"id": "a", "text": "one"}, {"id": "b", "text": "two"}]
    out = tmp_path / "preds.jsonl"
    model = OmniEdu("stub-model", StubBackend(fail_on=("one",)), concurrency=1)
    summary = model.batch(rows, out=out)
    assert summary["error"] == 1 and summary["ok"] == 1
    records = {record["id"]: record for record in load_jsonl(out)}
    assert "RuntimeError: boom" in records["a"]["error"]
    assert records["b"]["error"] is None


def test_batch_accepts_a_limit(tmp_path) -> None:
    rows = [{"id": str(index), "text": f"t{index}"} for index in range(5)]
    model = OmniEdu("stub-model", StubBackend(), concurrency=2)
    summary = model.batch(rows, out=tmp_path / "o.jsonl", limit=2)
    assert summary["total"] == 2 and summary["ok"] == 2


def test_sync_call_inside_a_loop_explains_itself() -> None:
    model = OmniEdu("stub-model", StubBackend())

    async def outer() -> None:
        with pytest.raises(RuntimeError, match="await"):
            model.chat(text="hi")

    asyncio.run(outer())


def test_sync_calls_reuse_one_loop_and_close_cleanly(tmp_path) -> None:
    """chat -> batch -> close must not leave the client on a dead loop."""
    backend = StubBackend()
    model = OmniEdu("stub-model", backend, concurrency=2)
    assert model.chat(text="one").text == "echo"
    summary = model.batch([{"id": "a", "text": "two"}], out=tmp_path / "o.jsonl")
    assert summary["ok"] == 1
    model.close()
    model.close()  # idempotent
