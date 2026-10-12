"""End-to-end checks against a throwaway OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from omniedu import OmniEdu


class _Handler(BaseHTTPRequestHandler):
    received: list[tuple[str, dict]] = []

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        _Handler.received.append((self.path, body))
        payload = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "test"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "42"},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:  # keep pytest output clean
        return None


@pytest.fixture()
def endpoint() -> str:
    _Handler.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("with_v1", [True, False])
def test_chat_request_matches_the_evaluation_protocol(endpoint: str, with_v1: bool) -> None:
    base_url = f"{endpoint}/v1" if with_v1 else endpoint
    model = OmniEdu.from_pretrained("4B", base_url=base_url, served_model="omniedu-4b")
    try:
        response = model.chat(text="hi", task="solve_reasoned", max_new_tokens=512)
    finally:
        model.close()

    path, body = _Handler.received[-1]
    assert path == "/v1/chat/completions"  # never /v1/v1/...
    assert body["model"] == "omniedu-4b"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 512
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert response.text == "42"
    assert response.finish_reason == "stop"


def test_images_are_sent_as_content_parts(endpoint: str) -> None:
    import base64

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    model = OmniEdu.from_pretrained("4B", base_url=endpoint, served_model="m")
    try:
        model.chat(text="look <image> here", image=png)
    finally:
        model.close()

    content = _Handler.received[-1][1]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text", "image_url", "text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_batch_against_the_endpoint(endpoint: str, tmp_path) -> None:
    out = tmp_path / "preds.jsonl"
    model = OmniEdu.from_pretrained("4B", base_url=endpoint, served_model="m", concurrency=2)
    try:
        summary = model.batch([{"id": "1", "text": "a"}, {"id": "2", "text": "b"}], out=out)
    finally:
        model.close()

    assert summary["ok"] == 2 and summary["error"] == 0
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {line["id"] for line in lines} == {"1", "2"}
    assert all(line["response"] == "42" for line in lines)


def test_short_alias_sends_the_name_omniedu_serve_prints(endpoint: str) -> None:
    """The README's `omniedu serve --model 4B` prints `--served-model-name
    omniedu-4b`; the README's `from_pretrained("4B", base_url=...)` must send
    exactly that name, or the two examples cannot talk to each other."""
    model = OmniEdu.from_pretrained("4B", base_url=endpoint)
    try:
        model.chat(text="hi")
    finally:
        model.close()

    assert _Handler.received[-1][1]["model"] == "omniedu-4b"


def test_served_model_overrides_the_derived_name(endpoint: str) -> None:
    model = OmniEdu.from_pretrained("27B", base_url=endpoint, served_model="my-server")
    try:
        model.chat(text="hi")
    finally:
        model.close()

    assert _Handler.received[-1][1]["model"] == "my-server"


def test_full_hub_id_is_sent_unchanged(endpoint: str) -> None:
    """A server started without `--served-model-name` answers to the Hub id, so an
    explicit id must not be rewritten into `omniedu-...`."""
    model = OmniEdu.from_pretrained("OpenDCAI/Omni-Edu-9B", base_url=endpoint)
    try:
        model.chat(text="hi")
    finally:
        model.close()

    assert _Handler.received[-1][1]["model"] == "OpenDCAI/Omni-Edu-9B"
