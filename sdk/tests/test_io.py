import json

import pytest

from omniedu.io import (
    Request,
    append_jsonl,
    compact_jsonl,
    iter_jsonl,
    load_jsonl,
    normalise_request,
)


def test_sharegpt_row_is_normalised() -> None:
    request = normalise_request(
        {
            "id": "r1",
            "messages": [
                {"role": "system", "content": "row system"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "images": ["a.png"],
        }
    )
    assert isinstance(request, Request)
    assert request.id == "r1"
    assert request.system == "row system"
    assert [message["role"] for message in request.messages] == ["user", "assistant"]
    assert request.images == ["a.png"]


def test_flat_row_is_normalised() -> None:
    request = normalise_request({"question": "2+2?", "image": "b.png"}, index=7)
    assert request.messages == [{"role": "user", "content": "2+2?"}]
    assert request.images == ["b.png"]
    assert request.id == "row-000007"


def test_defaults_are_applied_and_overridden() -> None:
    assert normalise_request({"text": "x"}, task="solve_reasoned").task == "solve_reasoned"
    assert normalise_request({"text": "x", "task": "own"}, task="solve_reasoned").task == "own"
    assert normalise_request({"text": "x"}, system="sys").system == "sys"


def test_row_without_text_or_messages_is_rejected() -> None:
    with pytest.raises(ValueError):
        normalise_request({"id": "empty"})


def test_round_trip_jsonl(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"id": "a"})
    append_jsonl(path, {"id": "b"})
    assert [row["id"] for row in iter_jsonl(path)] == ["a", "b"]
    assert len(load_jsonl(path)) == 2


def test_compact_drops_duplicates_and_unkept_rows(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    for record in (
        {"id": "a", "response": "old"},
        {"id": "b", "error": "RuntimeError: boom"},
        {"id": "a", "response": "new"},
    ):
        append_jsonl(path, record)
    removed = compact_jsonl(path, key="id", keep={"a"})
    assert removed == 2
    rows = load_jsonl(path)
    assert rows == [{"id": "a", "response": "new"}]


def test_compact_is_a_no_op_when_already_compact(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    append_jsonl(path, {"id": "a"})
    before = path.read_text(encoding="utf-8")
    assert compact_jsonl(path, key="id", keep={"a"}) == 0
    assert path.read_text(encoding="utf-8") == before
    assert compact_jsonl(tmp_path / "missing.jsonl", key="id", keep=set()) == 0


def test_append_jsonl_keeps_unicode_readable(tmp_path) -> None:
    path = tmp_path / "u.jsonl"
    append_jsonl(path, {"id": "x", "response": "答案"})
    raw = path.read_text(encoding="utf-8")
    assert "答案" in raw
    assert json.loads(raw)["response"] == "答案"
