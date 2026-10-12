"""Aggregate a results tree into the paper's three main tables (Markdown).

Input is what ``omniedu eval`` (and, for the judged columns, ``omniedu judge``)
writes under a results root::

    <results-root>/<model>/<Benchmark>/summary.json      generation + deterministic scores
    <results-root>/<model>/<Benchmark>/judge/<task>.jsonl    judge workload (the inputs)
    <results-root>/<model>/<Benchmark>/judge/*.judgments.jsonl  judge outputs

The research tree keeps the same per-benchmark ``summary.json`` but writes the
judge products elsewhere; :func:`find_judgments` accepts those layouts too, so an
existing run can be summarised without re-running anything.

Columns follow the paper's three main tables:

* curriculum grounding - K12-Bench EM / F1, MathFish Acc., EDUMATH MaC
* K-12 problem solving - GAOKAO-Bench Full, EXAMS-V Overall, MDK12-Bench Full
* pedagogical tutoring - MathTutor-S / -SH, TutorBench, LongTutor-E / -T

Aggregation rules are the ones the paper used:

* GAOKAO ``Full`` = (objective correct + subjective score) / (objective max + subjective max)
* MDK12 ``Full`` = (objective accuracy x objective count + mean open score x open count) / total
* TutorBench ``Overall`` = mean over every rubric value that was judged
* LongTutor ``Evidence`` = mean of the IE / MR / HC judge accuracies
* LongTutor ``Teaching`` = the judge's own ``overall_score`` (not the mean of the shown columns)

A cell with no source prints as ``-`` and is listed under "missing" in the JSON
report, so a partial run never looks like a complete one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .io import iter_jsonl

#: (column, key, formatter).  Percent columns are stored as fractions and scaled
#: on print; ``LongTutor-T`` stays on the rubric's own 1-5 scale.
Table = tuple[str, ...]

CURRICULUM: Table = ("K12-Bench EM", "K12-Bench F1", "MathFish Acc.", "EDUMATH MaC")
PROBLEM_SOLVING: Table = ("GAOKAO-Bench Full", "EXAMS-V Overall", "MDK12-Bench Full")
TUTORING: Table = ("MathTutor-S", "MathTutor-SH", "TutorBench", "LongTutor-E", "LongTutor-T")

#: Metrics we produce that are *not* one of the paper's columns.  They are
#: appended to one table only, and only when some model has a value.
CURRICULUM_EXTRA: Table = ("EDUMATH GOOD",)
PROBLEM_SOLVING_EXTRA: Table = ("GAOKAO-Bench Obj", "MDK12-Bench Obj")
TUTORING_EXTRA: Table = ()

_PERCENT = {
    "K12-Bench EM",
    "K12-Bench F1",
    "MathFish Acc.",
    "EDUMATH MaC",
    "EDUMATH GOOD",
    "GAOKAO-Bench Obj",
    "MDK12-Bench Obj",
    "GAOKAO-Bench Full",
    "EXAMS-V Overall",
    "MDK12-Bench Full",
    "MathTutor-S",
    "MathTutor-SH",
    "TutorBench",
    "LongTutor-E",
}

MISSING = "-"

#: Main-table column -> (benchmark, appendix column).  ``summary.json`` carries a
#: ``metrics`` block in the appendix's names; this maps it onto the three tables.
FROM_METRICS: dict[str, tuple[str, str]] = {
    "K12-Bench EM": ("K12-Bench", "Overall EM"),
    "K12-Bench F1": ("K12-Bench", "Overall F1"),
    "MathFish Acc.": ("MathFish", "Overall Acc"),
    "EDUMATH MaC": ("EDUMATH", "MaC"),
    "EDUMATH GOOD": ("EDUMATH", "Classifier GOOD"),
    "GAOKAO-Bench Full": ("GAOKAO-Bench", "Full Rate"),
    "GAOKAO-Bench Obj": ("GAOKAO-Bench", "Obj Rate"),
    "EXAMS-V Overall": ("EXAMS-V", "Overall"),
    "MDK12-Bench Full": ("MDK12-Bench", "Full Overall"),
    "MDK12-Bench Obj": ("MDK12-Bench", "Objective Avg"),
    "MathTutor-S": ("MathTutorBench", "Scaffold WR"),
    "MathTutor-SH": ("MathTutorBench", "Scaffold-hard WR"),
    "TutorBench": ("TutorBench", "Overall"),
    "LongTutor-E": ("LongTutor", "Evidence Avg."),
    "LongTutor-T": ("LongTutor", "Teaching Avg."),
}


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for row in iter_jsonl(path):
        return row
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if value is not None]
    return sum(items) / len(items) if items else None


# --------------------------------------------------------------------------- judgments

JUDGE_OUTPUT_GLOBS = (
    "{model}/{bench}/judge/*.judgments.jsonl",
    "{model}/{bench}/*.judgments.jsonl",
    "{model}/subjective_handoff/bundle/judge_outputs*/*.judgments.jsonl",
)


def find_judgments(
    model_dir: Path, bench: str, extra_roots: Sequence[Path]
) -> list[dict[str, Any]]:
    """Return the judge records for one (model, benchmark); first id wins."""
    files: list[Path] = []
    for pattern in JUDGE_OUTPUT_GLOBS:
        files += sorted(model_dir.glob(pattern.format(model=model_dir.name, bench=bench)))
    for root in extra_roots:
        files += sorted(root.glob(f"{model_dir.name}/*.judgments.jsonl"))
    records: dict[str, dict[str, Any]] = {}
    for path in files:
        for row in iter_jsonl(path):
            if row.get("judge_id") and not row.get("error"):
                records.setdefault(row["judge_id"], row)
    return list(records.values())


def find_workload(
    model_dir: Path, bench: str, task: str, extra_roots: Sequence[Path]
) -> dict[str, dict[str, Any]]:
    """Judge *inputs* keyed by ``judge_id`` (needed for score maxima, categories)."""
    files = sorted(model_dir.glob(f"{bench}/judge/{task}.jsonl"))
    files += sorted(model_dir.glob(f"subjective_handoff/bundle/data/{bench}/*/*.jsonl"))
    for root in extra_roots:
        files += sorted(root.glob(f"{model_dir.name}/*.jsonl"))
    rows: dict[str, dict[str, Any]] = {}
    for path in files:
        for row in iter_jsonl(path):
            if row.get("judge_id"):
                rows.setdefault(row["judge_id"], row)
    return rows


# --------------------------------------------------------------------------- per-benchmark


def k12(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    summary = _json(model_dir / "K12-Bench" / "summary.json") or {}
    overall = summary.get("overall") or summary
    metrics["K12-Bench EM"] = _number(overall.get("exact_match"))
    metrics["K12-Bench F1"] = _number(overall.get("f1"))


def mathfish(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    """The official ``scripts/eval.py`` output is authoritative; the runner's own
    ``summary.json`` only carries an informational accuracy."""
    official = sorted((model_dir / "MathFish" / "official_eval").glob("results_*.jsonl"))
    if official:
        metrics["MathFish Acc."] = _number((_first_jsonl(official[0]) or {}).get("acc"))
    if metrics.get("MathFish Acc.") is None:
        runner = next(iter(sorted((model_dir / "MathFish").glob("*.summary.json"))), None)
        metrics["MathFish Acc."] = (
            _number((_json(runner) or {}).get("accuracy")) if runner else None
        )


def gaokao(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    objective = (
        _json(model_dir / "GAOKAO-Bench" / "objective" / "result" / "correction_score.json") or {}
    )
    objective_score = _number(objective.get("correct_score"))
    objective_max = _number(objective.get("total_score"))
    # Prefer the raw marks: the official scorer's own ``scoring_rate`` is rounded
    # to two decimals and loses ~0.02 of a point against the paper's table.
    objective_rate = None
    if objective_score is not None and objective_max:
        objective_rate = objective_score / objective_max
    if objective_rate is None:
        objective_rate = _number(objective.get("scoring_rate"))
    metrics["GAOKAO-Bench Obj"] = objective_rate

    workload = find_workload(model_dir, "GAOKAO-Bench", "gaokao_subjective", extra)
    judged = find_judgments(model_dir, "GAOKAO-Bench", extra)
    subjective_score = subjective_max = 0.0
    for record in judged:
        score = _number((record.get("parsed") or {}).get("score"))
        maximum = _number((workload.get(record["judge_id"]) or {}).get("score"))
        if score is None or not maximum:
            continue
        subjective_score += score
        subjective_max += maximum

    if objective_score is None or not objective_max or not subjective_max:
        return
    metrics["GAOKAO-Bench Full"] = (objective_score + subjective_score) / (
        objective_max + subjective_max
    )


def exams_v(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    for name in ("summary.json", "predictions.summary.json"):
        summary = _json(model_dir / "EXAMS-V" / name)
        if summary and _number(summary.get("accuracy")) is not None:
            metrics["EXAMS-V Overall"] = _number(summary.get("accuracy"))
            return


def mdk12(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    summary = _json(model_dir / "MDK12-Bench" / "summary.json") or {}
    objective_avg = _number(summary.get("objective_accuracy"))
    objective_count = _number(summary.get("objective_scored"))
    open_count = _number(summary.get("judge_pending")) or 0
    metrics["MDK12-Bench Obj"] = objective_avg

    workload = find_workload(model_dir, "MDK12-Bench", "mdk12_open", extra)
    judged = find_judgments(model_dir, "MDK12-Bench", extra)
    open_scores = [
        score
        for score in (_number((record.get("parsed") or {}).get("score")) for record in judged)
        if score is not None
    ]
    open_mean = _mean(open_scores)
    if objective_avg is not None and objective_count and open_mean is not None:
        if not open_count:  # older summaries do not record the open row count
            open_count = float(len(workload) or len(open_scores))
        metrics["MDK12-Bench Full"] = (objective_avg * objective_count + open_mean * open_count) / (
            objective_count + open_count
        )


def edumath(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    """``EDUMATH MaC`` is the judge's Yes-rate; ``EDUMATH GOOD`` is the official
    classifier's rate.  They are different metrics and are kept separate."""
    judged = find_judgments(model_dir, "EDUMATH", extra)
    labels = [
        label
        for label in ((record.get("parsed") or {}).get("label") for record in judged)
        if label in (0, 1)
    ]
    if labels:
        metrics["EDUMATH MaC"] = sum(labels) / len(labels)
    classifier = _json(
        model_dir / "EDUMATH-standards-aligned" / "classifier" / "classified_official.summary.json"
    )
    if classifier:
        metrics["EDUMATH GOOD"] = _number(classifier.get("good_rate"))


def mathtutorbench(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    for column, task in (
        ("MathTutor-S", "scaffolding_generation"),
        ("MathTutor-SH", "scaffolding_generation_hard"),
    ):
        summary = (
            _json(model_dir / "MathTutorBench" / task / "official_reward" / "summary.json") or {}
        )
        metrics[column] = _number((summary.get("result") or {}).get("win_rate"))


def tutorbench(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    judged = find_judgments(model_dir, "TutorBench", extra)
    values: list[float] = []
    for record in judged:
        criteria = (record.get("parsed") or {}).get("criteria")
        if not isinstance(criteria, list):
            continue
        for item in criteria:
            if isinstance(item, dict) and item.get("pass") is not None:
                values.append(1.0 if item["pass"] else 0.0)
    metrics["TutorBench"] = _mean(values)


def longtutor(metrics: dict[str, Any], model_dir: Path, extra: Sequence[Path]) -> None:
    judged = find_judgments(model_dir, "LongTutor", extra)
    per_type = ([], [], [])
    overall: list[float] = []
    for record in judged:
        memory = record.get("memory_correct")
        if isinstance(memory, list):
            for index in range(3):
                if index < len(memory) and isinstance(memory[index], bool):
                    per_type[index].append(1.0 if memory[index] else 0.0)
        score = _number((record.get("content_scores") or {}).get("overall_score"))
        if score is not None:
            overall.append(score)
    metrics["LongTutor-E"] = _mean(_mean(values) for values in per_type)
    metrics["LongTutor-T"] = _mean(overall)


COLLECTORS: dict[str, Callable[[dict[str, Any], Path, Sequence[Path]], None]] = {
    "k12": k12,
    "mathfish": mathfish,
    "gaokao": gaokao,
    "exams_v": exams_v,
    "mdk12": mdk12,
    "edumath": edumath,
    "mathtutorbench": mathtutorbench,
    "tutorbench": tutorbench,
    "longtutor": longtutor,
}


@dataclass
class Model:
    name: str
    group: str = ""
    size: str = ""
    display: str = ""
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.display or self.name


# --------------------------------------------------------------------------- rendering


def _cell(column: str, value: float | None) -> str:
    if value is None:
        return MISSING
    return f"{value * 100:.2f}%" if column in _PERCENT else f"{value:.2f}"


def _table(columns: Table, models: Sequence[Model], extras: Table = ()) -> str:
    extra = tuple(
        column
        for column in extras
        if column not in columns and any(model.metrics.get(column) is not None for model in models)
    )
    columns = columns + extra
    header = "| Model | Size | " + " | ".join(columns) + " |"
    divider = "|---|---:|" + "---:|" * len(columns)
    lines = [header, divider]
    group = None
    for model in models:
        if model.group and model.group != group:
            group = model.group
            lines.append(f"| **{group}** |  | " + " | ".join("" for _ in columns) + " |")
        cells = " | ".join(_cell(column, model.metrics.get(column)) for column in columns)
        lines.append(f"| {model.label} | {model.size or MISSING} | {cells} |")
    return "\n".join(lines)


def render(models: Sequence[Model]) -> str:
    blocks = [
        "### Curriculum grounding\n\n" + _table(CURRICULUM, models, CURRICULUM_EXTRA),
        "### K-12 problem solving\n\n" + _table(PROBLEM_SOLVING, models, PROBLEM_SOLVING_EXTRA),
        "### Pedagogical tutoring\n\n" + _table(TUTORING, models, TUTORING_EXTRA),
    ]
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------- entry point


def collect(
    results_roots: Sequence[Path],
    judgment_roots: Sequence[Path],
    labels: dict[str, dict[str, str]],
) -> list[Model]:
    models: dict[str, Model] = {}
    for root in results_roots:
        if not root.is_dir():
            raise SystemExit(f"results root not found: {root}")
        for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            model = models.setdefault(model_dir.name, Model(name=model_dir.name))
            for column, collector in COLLECTORS.items():
                collector(model.metrics, model_dir, judgment_roots)
            # A summary written by ``omniedu eval`` already carries the appendix
            # table columns; use them when they are present.
            for column, (benchmark, key) in FROM_METRICS.items():
                summary = _json(model_dir / benchmark / "summary.json") or {}
                value = (summary.get("metrics") or {}).get(key)
                if value is not None:
                    model.metrics[column] = value / 100 if column in _PERCENT else value
    for name, extra in labels.items():
        model = models.setdefault(name, Model(name=name))
        model.display = extra.get("display", "")
        model.size = extra.get("size", "")
        model.group = extra.get("group", "")
    return sorted(models.values(), key=lambda item: (item.group, item.name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omniedu report", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-root",
        action="append",
        required=True,
        type=Path,
        help="directory whose children are per-model result dirs (repeatable)",
    )
    parser.add_argument(
        "--judgments-root",
        action="append",
        default=[],
        type=Path,
        help="extra root holding <model>/*.judgments.jsonl (repeatable)",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help='JSON: {"<model dir>": {"display": ..., "size": ..., "group": ...}}',
    )
    parser.add_argument("--out", type=Path, help="write Markdown here (default: stdout)")
    parser.add_argument("--json", type=Path, help="also write the collected metrics as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else {}
    models = collect(args.results_root, args.judgments_root, labels)
    markdown = render(models)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(markdown)

    missing = {
        model.name: [
            column
            for column in CURRICULUM + PROBLEM_SOLVING + TUTORING
            if model.metrics.get(column) is None
        ]
        for model in models
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"models": {m.name: m.metrics for m in models}, "missing": missing},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {"models": len(models), "incomplete": {k: v for k, v in missing.items() if v}},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
