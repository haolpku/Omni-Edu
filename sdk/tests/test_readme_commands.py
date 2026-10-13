"""Every `omniedu ...` command printed in the READMEs must actually exist.

The docs are part of the product, so a flag that was renamed or a subcommand that
never got written should fail here rather than at a user's terminal.  The check
parses the shell blocks instead of running them: it asserts every subcommand and
every ``--flag`` appears in that command's own ``--help``.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from argparse import _SubParsersAction

import pytest

from omniedu.cli import build_parser as core_parser
from omniedu.eval import benchmarks as catalogue
from omniedu.eval.cli import build_parser as eval_parser
from omniedu.eval.judge.cli import build_parser as judge_parser
from omniedu.eval.report import build_parser as report_parser

ROOT = Path(__file__).resolve().parents[1]
READMES = tuple(name for name in ("README.md", "README.zh.md") if (ROOT / name).is_file())

# The READMEs are part of the source tree, not the installed package, so this
# module has nothing to check when it is run against a copy of `src/` alone.
pytestmark = pytest.mark.skipif(not READMES, reason="READMEs are not present (not a source tree)")


def _usage() -> dict[str, str]:
    """Subcommand name -> its own ``--help`` text."""
    parser = core_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, _SubParsersAction))
    usage = {name: child.format_help() for name, child in subparsers.choices.items()}
    # These three own large parsers of their own and are dispatched before the
    # core parser is built (see omniedu.cli.main).
    usage["eval"] = eval_parser(catalogue.select("all")).format_help()
    usage["judge"] = judge_parser().format_help()
    usage["report"] = report_parser().format_help()
    return usage


def _subcommands() -> set[str]:
    parser = core_parser()
    subparsers = next(a for a in parser._actions if isinstance(a, _SubParsersAction))
    return set(subparsers.choices) | {"eval", "judge", "report"}


def _commands() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for name in READMES:
        text = (ROOT / name).read_text(encoding="utf-8")
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            # Join the backslash continuations first, then take one command per line.
            joined = re.sub(r"\\\n\s*", " ", block)
            for line in joined.splitlines():
                line = line.split("#")[0].strip()
                if line.startswith("omniedu "):
                    found.append((name, line))
    return found


def test_both_readmes_have_commands_to_check() -> None:
    assert len(_commands()) >= 6


def test_every_documented_command_exists() -> None:
    usage = _usage()
    known = _subcommands()
    problems = []
    for name, line in _commands():
        argv = shlex.split(line)
        subcommand = argv[1]
        if subcommand not in known:
            problems.append(f"{name}: `omniedu {subcommand}` does not exist")
            continue
        help_text = usage[subcommand]
        for token in argv[2:]:
            flag = token.split("=")[0]
            if flag.startswith("--") and flag not in help_text:
                problems.append(f"{name}: `{flag}` is not accepted by `omniedu {subcommand}`")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", READMES)
def test_readme_examples_do_not_reference_unshipped_paths(name: str) -> None:
    """The research tree is git-ignored, so a README must not point users at it."""
    text = (ROOT / name).read_text(encoding="utf-8")
    for stale in ("evaluation/PROTOCOL.md", "evaluation/README.md", "evaluation/judge"):
        assert stale not in text, f"{name} points at the research tree: {stale}"
