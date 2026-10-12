"""`omniedu eval` targets the same served name `omniedu serve` prints."""

from __future__ import annotations

import json

from omniedu.eval import cli


def test_eval_resolves_an_alias_to_the_served_name(capsys) -> None:
    code = cli.main(
        [
            "--benchmarks",
            "GPQA",
            "--model",
            "4B",
            "--base-url",
            "http://127.0.0.1:8000",
            "--dry-run",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == "omniedu-4b"
    assert plan["output_root"].endswith("4b")  # the tag keeps the short form


def test_eval_leaves_a_real_served_name_alone(capsys) -> None:
    code = cli.main(
        [
            "--benchmarks",
            "GPQA",
            "--model",
            "qwen3-8b",
            "--base-url",
            "http://127.0.0.1:8000/v1",
            "--dry-run",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model"] == "qwen3-8b"
    assert plan["base_urls"] == ["http://127.0.0.1:8000/v1"]


def test_list_prints_the_upstream_source(capsys) -> None:
    """The README tells users `--list` names the repository to clone."""
    assert cli.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "https://github.com/haolpku/K12-KGraph" in out
    assert "https://github.com/allenai/mathfish" in out
