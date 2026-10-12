"""The printed `vllm serve` command has to be shell-copy-pasteable."""

from __future__ import annotations

import shlex

from omniedu.cli import main


def test_serve_command_survives_a_shell_split(capsys) -> None:
    assert main(["serve", "--model", "27B", "--tensor-parallel-size", "4"]) == 0
    printed = capsys.readouterr().out.strip()
    argv = shlex.split(printed)  # what the user's shell would do before running it

    assert argv[0] == "vllm" and argv[1] == "serve"
    assert "--served-model-name" in argv
    assert argv[argv.index("--served-model-name") + 1] == "omniedu-27b"
    # The option's value carries a space; an unquoted join would split it in two.
    assert argv[argv.index("--limit-mm-per-prompt") + 1] == '{"image": 12}'


def test_serve_keeps_an_explicit_served_name(capsys) -> None:
    assert main(["serve", "--model", "4B", "--served-model", "my-name"]) == 0
    argv = shlex.split(capsys.readouterr().out.strip())
    assert argv[argv.index("--served-model-name") + 1] == "my-name"
