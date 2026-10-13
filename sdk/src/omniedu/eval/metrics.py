"""The per-benchmark metric columns, named and ordered like the paper's tables.

Each benchmark's ``summary.json`` carries a ``metrics`` block whose keys are the
corresponding appendix table's columns, in the appendix's order.  Rates are
percentages (0-100, the unit the tables print); the LongTutor rubric dimensions
stay on their own 1-5 scale.  A column the run did not produce is ``None``, so a
missing judged stage is visible instead of silently absent.

Two names are disambiguated because the appendix repeats them across grouped
headers: LongTutor's two *Avg.* columns become ``Evidence Avg.`` / ``Teaching
Avg.``, and K12-Bench's per-family *EM*/*F1* are prefixed with their family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import iter_jsonl

K12: tuple[str, ...] = (
    "Ground EM",
    "Ground F1",
    "Prereq EM",
    "Prereq F1",
    "Neighbor EM",
    "Neighbor F1",
    "Evidence EM",
    "Evidence F1",
    "Locate EM",
    "Locate F1",
    "Overall EM",
    "Overall F1",
)
MATHFISH: tuple[str, ...] = (
    "Pos: Should Yes",
    "Neg: ATC Neighbor",
    "Neg: Same-domain Same-grade",
    "Neg: Same-domain Diff-grade",
    "Neg: Diff-domain Same-grade",
    "Neg: Diff-domain Diff-grade",
    "Overall Acc",
)
EDUMATH: tuple[str, ...] = ("Format Pass", "Classifier GOOD", "MaC")
GAOKAO: tuple[str, ...] = (
    "Obj Rate",
    "Subj Rate",
    "Full Rate",
    "Chinese",
    "English",
    "Math",
    "Physics",
    "Chemistry",
    "Biology",
    "Politics",
    "History",
    "Geography",
)
EXAMS_V: tuple[str, ...] = (
    "Overall",
    "Text",
    "Image+Text",
    "Natural Science",
    "Social Sciences",
    "Other",
    "Chinese",
    "English",
)
MDK12: tuple[str, ...] = (
    "Single-choice",
    "Multiple-choice",
    "True/False",
    "Objective Avg",
    "Open Correctness",
    "Full Overall",
)
MATHTUTOR: tuple[str, ...] = (
    "Scaffold WR",
    "Scaffold-hard WR",
    "Pedagogy WR",
    "Pedagogy-hard WR",
)
TUTORBENCH: tuple[str, ...] = (
    "UC1 Text",
    "UC1 Multimodal",
    "UC2 Text",
    "UC2 Multimodal",
    "UC3 Text",
    "UC3 Multimodal",
    "Text",
    "Multimodal",
    "Overall",
)
LONGTUTOR: tuple[str, ...] = (
    "IE",
    "MR",
    "HC",
    "Evidence Avg.",
    "Acc",
    "F1",
    "R-L",
    "Hist.",
    "Strat.",
    "Coher.",
    "Appr.",
    "Teaching Avg.",
)
IFEVAL: tuple[str, ...] = ("Prompt Strict", "Inst. Strict", "Prompt Loose", "Inst. Loose")
GPQA: tuple[str, ...] = ("Main Acc.", "Diamond Acc.")
MMMU_PRO: tuple[str, ...] = (
    "Overall",
    "Art",
    "Business",
    "Science",
    "Health",
    "Humanities",
    "Tech.",
)

#: Appendix table columns per benchmark, keyed the way ``benchmarks.json`` names them.
COLUMNS: dict[str, tuple[str, ...]] = {
    "K12-Bench": K12,
    "MathFish": MATHFISH,
    "EDUMATH": EDUMATH,
    "GAOKAO-Bench": GAOKAO,
    "EXAMS-V": EXAMS_V,
    "MDK12-Bench": MDK12,
    "MathTutorBench": MATHTUTOR,
    "TutorBench": TUTORBENCH,
    "LongTutor": LONGTUTOR,
    "IFEval": IFEVAL,
    "GPQA": GPQA,
    "MMMU-Pro": MMMU_PRO,
}

#: Columns whose value is a percentage.  Everything else is on the benchmark's
#: own scale (LongTutor's rubric dimensions).
PERCENT = frozenset(
    column
    for benchmark, columns in COLUMNS.items()
    if benchmark != "LongTutor"
    for column in columns
) | {"IE", "MR", "HC", "Evidence Avg.", "R-L"}

#: Domain key in MMMU-Pro's per-subject breakdown -> appendix column.
MMMU_DOMAINS = {
    "Art and Design": "Art",
    "Business": "Business",
    "Science": "Science",
    "Health and Medicine": "Health",
    "Humanities and Social Science": "Humanities",
    "Tech and Engineering": "Tech.",
}


def build(benchmark: str, values: dict[str, float | None]) -> dict[str, float | None]:
    """Order ``values`` into ``benchmark``'s table columns, blanking the rest."""
    return {column: values.get(column) for column in COLUMNS[benchmark]}


# --------------------------------------------------------------------------- judge side


def judgments(output_dir: Path) -> list[dict[str, Any]]:
    """Error-free judge records written next to this benchmark, keyed by id.

    A re-judged workload can carry the same ``judge_id`` twice with different
    answers.  The **first** record wins, which is the rule the paper's dedup
    step used, so the aggregates here match the reported numbers.
    """
    found: dict[str, dict[str, Any]] = {}
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*.judgments.jsonl")):
            for row in iter_jsonl(path):
                if row.get("judge_id") and not row.get("error"):
                    found.setdefault(row["judge_id"], row)
    return list(found.values())


def workload(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Judge *inputs* (``judge/<task>.jsonl``), keyed by ``judge_id``."""
    rows: dict[str, dict[str, Any]] = {}
    judge_dir = output_dir / "judge"
    if judge_dir.is_dir():
        for path in sorted(judge_dir.glob("*.jsonl")):
            if path.name.endswith(".judgments.jsonl"):
                continue
            for row in iter_jsonl(path):
                if row.get("judge_id"):
                    rows.setdefault(row["judge_id"], row)
    return rows


def _mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None]
    return sum(items) / len(items) if items else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent(value: float | None) -> float | None:
    return None if value is None else value * 100


def judged_longtutor(output_dir: Path) -> dict[str, float | None]:
    """Evidence, diagnosis and teaching columns that only the judge produces."""
    records = judgments(output_dir)
    per_type: tuple[list[float], list[float], list[float]] = ([], [], [])
    scores: dict[str, list[float]] = {}
    rouge: list[float] = []
    for record in records:
        memory = record.get("memory_correct")
        if isinstance(memory, list):
            for index in range(3):
                if index < len(memory) and isinstance(memory[index], bool):
                    per_type[index].append(1.0 if memory[index] else 0.0)
        for key, value in (record.get("content_scores") or {}).items():
            number = _number(value)
            if number is not None:
                scores.setdefault(key, []).append(number)
        number = _number(record.get("rouge_l"))
        if number is not None:
            rouge.append(number)
    typing = ("IE", "MR", "HC")
    columns: dict[str, float | None] = {
        name: _percent(_mean(values)) for name, values in zip(typing, per_type)
    }
    columns["Evidence Avg."] = _percent(_mean(_mean(values) for values in per_type))
    columns["R-L"] = _mean(rouge)
    for key, column in (
        ("history_score", "Hist."),
        ("strategy_score", "Strat."),
        ("coherence_score", "Coher."),
        ("appropriateness_score", "Appr."),
        ("overall_score", "Teaching Avg."),
    ):
        columns[column] = _mean(scores.get(key, []))
    return columns


def judged_tutorbench(output_dir: Path) -> dict[str, float | None]:
    records = judgments(output_dir)
    sources = workload(output_dir)
    buckets: dict[str, list[float]] = {}
    for record in records:
        criteria = (record.get("parsed") or {}).get("criteria")
        if not isinstance(criteria, list):
            continue
        values = [
            1.0 if item.get("pass") else 0.0
            for item in criteria
            if isinstance(item, dict) and item.get("pass") is not None
        ]
        if not values:
            continue
        batch = str((sources.get(record["judge_id"]) or {}).get("batch") or "")
        use_case = batch.split("_")[2] if batch.startswith("USE_CASE") else "1"
        modality = "Multimodal" if "MULTIMODAL" in batch.upper() else "Text"
        buckets.setdefault(f"UC{use_case} {modality}", []).extend(values)
        buckets.setdefault(modality, []).extend(values)
        buckets.setdefault("Overall", []).extend(values)
    return {column: _percent(_mean(buckets.get(column, []))) for column in TUTORBENCH}


def judged_mdk12(output_dir: Path) -> dict[str, float | None]:
    records = judgments(output_dir)
    scores = [
        score
        for score in (_number((record.get("parsed") or {}).get("score")) for record in records)
        if score is not None
    ]
    return {"Open Correctness": _percent(_mean(scores))}


def judged_edumath(output_dir: Path) -> dict[str, float | None]:
    labels = [
        label
        for label in ((record.get("parsed") or {}).get("label") for record in judgments(output_dir))
        if label in (0, 1)
    ]
    return {"MaC": _percent(_mean(labels))}


def judged_gaokao(output_dir: Path) -> dict[str, float | None]:
    """Subjective rate from the judged answers; the objective side is separate."""
    sources = workload(output_dir)
    got = total = 0.0
    for record in judgments(output_dir):
        score = _number((record.get("parsed") or {}).get("score"))
        maximum = _number((sources.get(record["judge_id"]) or {}).get("score"))
        if score is None or not maximum:
            continue
        got += score
        total += maximum
    return {"Subj Rate": _percent(got / total) if total else None}


JUDGED = {
    "LongTutor": judged_longtutor,
    "TutorBench": judged_tutorbench,
    "MDK12-Bench": judged_mdk12,
    "EDUMATH": judged_edumath,
    "GAOKAO-Bench": judged_gaokao,
}


def with_judged(
    benchmark: str, output_dir: Path, values: dict[str, float | None]
) -> dict[str, float | None]:
    """Merge the judged columns (when the judge has run) into a metrics block."""
    reader = JUDGED.get(benchmark)
    if reader is not None:
        for key, value in reader(output_dir).items():
            if value is not None:
                values[key] = value
    return build(benchmark, values)
