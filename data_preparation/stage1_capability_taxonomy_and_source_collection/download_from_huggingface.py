#!/usr/bin/env python3
"""Download stage-1 sources from the Hugging Face Hub.

The source list is intentionally left empty: the datasets used by Omni-Edu are
listed in the paper. Fill ``SOURCES`` in, or pass ``--source`` on the command
line, and the corresponding repositories are mirrored into this stage:

    python data_preparation/stage1_capability_taxonomy_and_source_collection/\
download_from_huggingface.py --source OpenDCAI/Omni-Edu=subject_competence/huggingface/omni-edu

Each ``--source`` takes ``<repo_id>=<destination>``; the destination is resolved
relative to this stage directory. GitHub-hosted sources need no helper here —
they are plain ``git clone`` targets, for example::

    git clone <url> data_preparation/stage1_capability_taxonomy_and_source_collection/\
<capability>/github/<repository>
"""

from __future__ import annotations

import argparse
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent

#: ``(repo_id, destination, repo_type)`` triples; fill in per capability.
SOURCES: list[tuple[str, str, str]] = [
    # ("<owner>/<dataset>", "<capability>/huggingface/<name>", "dataset"),
]


def download(repo_id: str, destination: str, repo_type: str, revision: str | None) -> Path:
    """Mirror one repository into the stage tree and return the local path."""
    from huggingface_hub import snapshot_download

    target = STAGE_DIR / destination
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    return target


def parse_source(value: str) -> tuple[str, str, str]:
    """Parse ``<repo_id>=<destination>`` (``=`` is split on the last occurrence)."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected <repo_id>=<destination>")
    repo_id, destination = value.rsplit("=", 1)
    repo_type = "dataset"
    if ":" in repo_id:
        repo_type, repo_id = repo_id.split(":", 1)
    return repo_id, destination, repo_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        default=[],
        metavar="REPO=DEST",
        help="repository to download; may be repeated",
    )
    parser.add_argument("--revision", default=None, help="branch, tag or commit")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    args = parser.parse_args()

    sources = list(args.source) or list(SOURCES)
    if not sources:
        parser.error("no sources configured: fill in SOURCES or pass --source")

    for repo_id, destination, repo_type in sources:
        if args.dry_run:
            print(f"[dry-run] {repo_id} ({repo_type}) -> {destination}")
            continue
        print(f"downloading {repo_id} ({repo_type}) -> {destination}", flush=True)
        download(repo_id, destination, repo_type, args.revision)


if __name__ == "__main__":
    main()
