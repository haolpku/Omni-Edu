"""Dataset directories and standard pool paths.

Stage 1 snapshots and stage 2 cleaning outputs share the same capability layout:

    data_preparation/stage1_.../<capability>/huggingface|github|direct/<source>
    data_preparation/stage2_.../<capability>/<dataset>/
    ├── cleaned/kept.jsonl          # standard cleaning pool (input of stage 2 dedup)
    ├── cleaned/rejected.jsonl      # dropped rows (optional)
    ├── cleaned/report.json         # per-dataset cleaning report (optional)
    ├── cleaned/media/…             # materialized images (if any)
    └── dedup/within/kept.jsonl     # within-dataset dedup output

`find_pools()` discovers datasets by ``<dataset>/cleaned/kept.jsonl``, so a dataset directory may sit
directly under a capability or one level deeper inside a source family, for example
``pedagogical_action/feat_foxglove_sefora/FEAT/cleaned/kept.jsonl``）。
"""

from __future__ import annotations

from pathlib import Path

DATA_PREPARATION = Path(__file__).resolve().parent.parent
STAGE1 = DATA_PREPARATION / "stage1_capability_taxonomy_and_source_collection"
STAGE2 = DATA_PREPARATION / "stage2_deterministic_cleaning_and_evaluation_decontamination"

#: the four capabilities stage 2 covers (general-purpose data is not one of them)
CAPABILITIES = (
    "subject_competence",
    "curriculum_grounding",
    "diagnostic_reasoning",
    "pedagogical_action",
)

STANDARD_POOL = Path("cleaned") / "kept.jsonl"
STANDARD_WITHIN_DEDUP = Path("dedup") / "within" / "kept.jsonl"


def source_dir(capability: str, dataset: str = "") -> Path:
    """Source directory of a capability (or of one dataset) in stage 1."""
    base = STAGE1 / capability
    return base / dataset if dataset else base


def dataset_dir(capability: str, dataset: str) -> Path:
    """Directory of one dataset in stage 2."""
    return STAGE2 / capability / dataset


def cleaned_dir(capability: str, dataset: str) -> Path:
    return dataset_dir(capability, dataset) / "cleaned"


def cleaned_path(capability: str, dataset: str) -> Path:
    """Standard cleaning output of a dataset (input of stage 2 dedup)."""
    return dataset_dir(capability, dataset) / STANDARD_POOL


def find_pools(root: Path | None = None) -> dict[str, Path]:
    """Dataset name → standard cleaning output, discovered by the canonical path only."""
    root = Path(root) if root is not None else STAGE2
    if root.name in CAPABILITIES:
        roots = [root]
    else:
        roots = [root / capability for capability in CAPABILITIES if (root / capability).is_dir()]
    found: dict[str, Path] = {}
    for base in roots:
        for pool in sorted(base.rglob(str(STANDARD_POOL))):
            found.setdefault(pool.parent.parent.name, pool)
    return found
