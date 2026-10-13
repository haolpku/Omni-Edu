"""The judged columns, checked against hand-computed aggregates.

No judge model is called.  The fixtures are fabricated ``*.judgments.jsonl``
records with numbers chosen so the expected mean, rate or bucket is obvious, so
what the test actually pins down is the *aggregation* -- the part that silently
decides what the paper's table reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniedu.eval import metrics, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def longtutor_record(index: int, memory: list[bool], overall: int, rouge: float) -> dict:
    return {
        "judge_id": f"longtutor::m::{index}",
        "memory_correct": memory,
        "content_scores": {
            "history_score": 1,
            "strategy_score": 2 if index == 0 else 1,
            "coherence_score": 3 if index == 0 else 1,
            "comprehension_score": 4,
            "appropriateness_score": 5 if index == 0 else 1,
            "overall_score": overall,
        },
        "rouge_l": rouge,
        "error": None,
    }


def test_longtutor_columns_are_the_paper_aggregates(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "judge" / "longtutor_teaching.judgments.jsonl",
        [
            longtutor_record(0, [True, False, True], overall=2, rouge=0.5),
            longtutor_record(1, [True, True, False], overall=1, rouge=0.1),
        ],
    )
    columns = metrics.judged_longtutor(tmp_path)

    assert columns["IE"] == 100.0  # both records answered the first query right
    assert columns["MR"] == 50.0
    assert columns["HC"] == 50.0
    assert columns["Evidence Avg."] == pytest.approx((100.0 + 50.0 + 50.0) / 3)
    assert columns["R-L"] == 0.3  # stayed on the 0-1 scale, not percent
    assert columns["Hist."] == 1.0
    assert columns["Strat."] == 1.5
    assert columns["Coher."] == 2.0
    assert columns["Appr."] == 3.0
    assert (
        columns["Teaching Avg."] == 1.5
    )  # the judge's overall_score, not an average of the shown columns


def test_longtutor_skips_failed_records(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "judge" / "longtutor_teaching.judgments.jsonl",
        [
            longtutor_record(0, [True, True, True], overall=5, rouge=1.0),
            {
                "judge_id": "longtutor::m::1",
                "memory_correct": None,
                "error": "RateLimitError: 429",
            },
        ],
    )
    columns = metrics.judged_longtutor(tmp_path)
    assert columns["IE"] == 100.0  # the failed row contributed nothing
    assert columns["Teaching Avg."] == 5.0


def test_tutorbench_buckets_by_use_case_and_modality(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "judge" / "tutorbench_rubric.jsonl",
        [
            {"judge_id": "tutorbench::m::1", "batch": "USE_CASE_1_TEXT"},
            {"judge_id": "tutorbench::m::2", "batch": "USE_CASE_1_MULTIMODAL"},
            {"judge_id": "tutorbench::m::3", "batch": "USE_CASE_2_TEXT"},
        ],
    )
    write_jsonl(
        tmp_path / "judge" / "tutorbench_rubric.judgments.jsonl",
        [
            {
                "judge_id": "tutorbench::m::1",
                "parsed": {"criteria": [{"pass": True}, {"pass": False}]},
            },
            {
                "judge_id": "tutorbench::m::2",
                "parsed": {"criteria": [{"pass": True}, {"pass": True}]},
            },
            {"judge_id": "tutorbench::m::3", "parsed": {"criteria": [{"pass": False}]}},
        ],
    )
    columns = metrics.judged_tutorbench(tmp_path)

    assert columns["UC1 Text"] == 50.0
    assert columns["UC1 Multimodal"] == 100.0
    assert columns["UC2 Text"] == 0.0
    assert columns["UC2 Multimodal"] is None
    assert columns["Text"] == pytest.approx(100 / 3)  # (1 + 0 + 0) / 3
    assert columns["Multimodal"] == 100.0
    assert columns["Overall"] == 60.0  # (1 + 0 + 1 + 1 + 0) / 5


def test_mdk12_open_correctness_is_the_mean_score(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "judge" / "mdk12_open.judgments.jsonl",
        [
            {"judge_id": "mdk12::m::1", "parsed": {"score": 1.0}},
            {"judge_id": "mdk12::m::2", "parsed": {"score": 0.5}},
        ],
    )
    assert metrics.judged_mdk12(tmp_path)["Open Correctness"] == 75.0


def test_gaokao_subjective_rate_uses_the_workload_maximum(tmp_path: Path) -> None:
    """The rate is marks-weighted, not a plain average over answers."""
    write_jsonl(
        tmp_path / "judge" / "gaokao_subjective.jsonl",
        [
            {"judge_id": "gaokao::m::1", "score": 10},
            {"judge_id": "gaokao::m::2", "score": 20},
        ],
    )
    write_jsonl(
        tmp_path / "judge" / "gaokao_subjective.judgments.jsonl",
        [
            {"judge_id": "gaokao::m::1", "parsed": {"score": 8.0}},
            {"judge_id": "gaokao::m::2", "parsed": {"score": 10.0}},
        ],
    )
    assert metrics.judged_gaokao(tmp_path)["Subj Rate"] == 60.0  # 18 / 30


def test_edumath_mac_is_the_yes_rate(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "judge" / "edumath_quality.judgments.jsonl",
        [
            {"judge_id": f"edumath::m::{index}", "parsed": {"label": label}}
            for index, label in enumerate([1, 1, 0, 1])
        ],
    )
    assert metrics.judged_edumath(tmp_path)["MaC"] == 75.0


def test_build_keeps_the_appendix_order_and_blanks_missing_columns() -> None:
    block = metrics.build("GPQA", {"Main Acc.": 34.15})
    assert list(block) == list(metrics.COLUMNS["GPQA"])
    assert block == {"Main Acc.": 34.15, "Diamond Acc.": None}
    for benchmark, columns in metrics.COLUMNS.items():
        assert list(metrics.build(benchmark, {})) == list(columns), benchmark


def test_rouge_l_is_rouge_score_with_the_default_tokenizer() -> None:
    """The R-L column is the upstream call, not a lexical-overlap improvement."""
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    from omniedu.eval.judge.judge_longtutor import rouge_l

    gold = "小同学别灰心，你上周在解决[2412]这类题目时可是算得又快又准呢"
    pred = "别灰心，你上周做[2412]的时候算得又快又准"
    expected = float(
        rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(gold, pred)["rougeL"].fmeasure
    )
    assert rouge_l(gold, pred) == expected
    assert rouge_l(gold, "") == 0.0
    assert rouge_l("", pred) == 0.0


def test_a_summary_without_a_judge_keeps_the_judged_columns_empty(tmp_path: Path) -> None:
    (tmp_path / "judge").mkdir()
    block = metrics.with_judged("MDK12-Bench", tmp_path, {"Objective Avg": 23.06})
    assert block["Objective Avg"] == 23.06
    assert block["Open Correctness"] is None
    assert block["Full Overall"] is None


def test_report_prefers_the_summary_metrics_block(tmp_path: Path) -> None:
    model = tmp_path / "some-model"
    (model / "K12-Bench").mkdir(parents=True)
    (model / "K12-Bench" / "summary.json").write_text(
        json.dumps(
            {
                "benchmark": "K12-Bench",
                "metrics": {"Overall EM": 63.12, "Overall F1": 76.69, "Ground EM": 58.0},
            }
        ),
        encoding="utf-8",
    )
    models = report.collect([tmp_path], [], {})
    assert [m.name for m in models] == ["some-model"]
    assert models[0].metrics["K12-Bench EM"] == 0.6312  # stored as a fraction for the table
    assert models[0].metrics["K12-Bench F1"] == 0.7669

    markdown = report.render(models)
    header = "| Model | Size | K12-Bench EM | K12-Bench F1 | MathFish Acc. | EDUMATH MaC |"
    assert header in markdown
    assert "| some-model | - | 63.12% | 76.69% | - | - |" in markdown
