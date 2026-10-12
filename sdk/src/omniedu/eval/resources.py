"""Locate data files that ship inside the package.

The runners need a few static assets that are ours rather than upstream's: the
vendored IFEval scorer, the judge prompts, the decontamination keep-sets and the
MathTutorBench overrides.  They live under ``omniedu/eval/`` and are read through
``importlib.resources`` so they resolve the same way from a wheel or a checkout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

PACKAGE = "omniedu.eval"


def package_path(*parts: str) -> Path:
    return Path(resources.files(PACKAGE).joinpath(*parts))


def judge_prompt(name: str) -> Path:
    return package_path("judge", "prompts", name)


def clean_keep_ids(benchmark: str) -> Path:
    return package_path("clean", benchmark, "keep_ids.json")


def ifeval_vendor() -> Path:
    return package_path("vendor", "ifeval_official")


def mathtutorbench_data(*parts: str) -> Path:
    return package_path("mathtutorbench", *parts)
