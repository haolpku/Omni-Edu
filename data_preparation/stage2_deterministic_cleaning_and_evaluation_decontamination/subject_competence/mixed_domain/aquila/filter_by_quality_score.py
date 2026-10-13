#!/usr/bin/env python3
"""Filter AquilaEdu after Stage-1 exact dedup.

Policy confirmed by user:
- Drop/unroute all records without quality scores.
- Drop quality < 4.0.
- Remove high-confidence mathematics because specialist math corpora are
  already abundant.
- Keep non-math and math-uncertain records for later classification/cleaning.

No source data is physically deleted; every bucket is written separately.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in sys.path:
    sys.path.insert(0, str(_COMMON_ROOT))
from common import cleaning, pools

from typing import Any

PIPE = pools.STAGE2 / "subject_competence"
OUT = PIPE / "AquilaEdu/quality_filtered"
SOURCE = pools.source_dir("subject_competence") / "huggingface" / "AquilaEdu"
DATE = "2026-08-05"
QUALITY_THRESHOLD = 4.0


MATH_STRONG_RE = re.compile(
    r"""
    (\\frac|\\sqrt|\\sum|\\prod|\\int|\\log|\\sin|\\cos|\\tan|
     \\begin\{|\\boxed|\\overline|\\angle|\\triangle|\\pi\b|
     \^\{|\_\{|[a-zA-Z]\s*\([^)]*\)\s*=|
     \b(?:solve|calculate|compute|evaluate|simplify|factorize|factor|
     polynomial|quadratic|equation|inequality|integer|prime number|
     probability|permutation|combination|derivative|integral|matrix|
     vector|geometry|algebra|arithmetic|fraction|decimal|percentage|
     perimeter|circumference|volume|area of|how many ways)\b|
     (?:求|计算|解|化简|因式分解|证明).{0,20}
     (?:值|方程|不等式|函数|面积|周长|体积|概率|导数|积分|矩阵|向量)|
     (?:方程|不等式|函数|数列|几何|代数|概率|排列组合|导数|积分|
     矩阵|向量|圆|三角形|正方形|长方形|梯形|圆柱|圆锥|质数|
     因数|倍数|分数|小数|百分数|周长|面积|体积))
    """,
    re.I | re.X,
)

MATH_UNIT_RE = re.compile(
    r"(?:多少|几种|几个|几次|计算).{0,30}(?:米|千米|厘米|分米|平方米|平方厘米|"
    r"立方米|立方厘米|公顷|元|本|个|棵|人|小时|分钟|天|千克|"
    r"公斤|克|升|毫升|吨|度|百分之几|倍|页|辆|只|张|块|盒|颗)",
    re.I,
)

ZH_WORD_PROBLEM_RE = re.compile(
    r"(?:多少|几种|几个|几次|计算|一共|总共|剩下|还剩|平均|每(?:个|人|天|小时|"
    r"分钟|米|千米|本|只|辆|箱|盒|瓶|组|排)|分成|分给|占.{0,8}(?:几分之几|"
    r"百分之几)|扩大|缩小|倍|速度|路程|单价|总价).{0,100}"
    r"(?:多少|几种|几个|几次|值|米|千米|厘米|分米|平方米|立方米|元|本|个|"
    r"棵|人|小时|分钟|天|千克|公斤|克|升|吨|百分之几|倍|页|辆|只|"
    r"张|块|盒|颗|组|排|方法|方式)",
    re.I | re.S,
)

EN_WORD_PROBLEM_RE = re.compile(
    r"\b(?:how many|how much|what is|find|determine)\b.{0,160}"
    r"\b(?:total|altogether|remaining|left|cost|price|distance|speed|"
    r"time|ratio|percent|average|number|amount|length|width|height|"
    r"area|volume|probability)\b",
    re.I | re.S,
)

NON_MATH_CONTEXT_RE = re.compile(
    r"\b(?:passage|article|according to|author|paragraph|story|"
    r"mainly about|best title|infer from|the text)\b|"
    r"(?:阅读(?:下面|材料|文章)|根据(?:文章|材料|短文)|文中|作者|段落|"
    r"主旨|标题|小说|诗歌|古诗|文言文)",
    re.I,
)

OPTION_RE = re.compile(r"(?:Options?|选项|选择)\s*[:：]|\([A-H]\)|[A-H][.、．]", re.I)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def classify_math(row: dict[str, Any]) -> tuple[str, list[str]]:
    instruction = str(row.get("instruction") or "")
    problem = str(row.get("input") or row.get("problem") or instruction)
    output = str(row.get("output") or "")
    answer = str(row.get("answer") or "").strip()
    text = f"{instruction}\n{problem}"
    reasons: list[str] = []
    reading_context = bool(NON_MATH_CONTEXT_RE.search(text))
    has_options = bool(OPTION_RE.search(problem))
    long_mc_context = has_options and len(problem) >= 500

    strong = MATH_STRONG_RE.search(text)
    # number, etc. Only trust strong patterns there when they are formal math
    # notation rather than a lone lexical keyword.
    formal_notation = bool(
        re.search(
            r"\\frac|\\sqrt|\\sum|\\prod|\\int|\\begin\{|\\boxed|"
            r"\^\{|\_\{|[a-zA-Z]\s*\([^)]*\)\s*=",
            text,
            re.I,
        )
    )
    if strong and (not has_options or formal_notation):
        reasons.append(f"strong_math_pattern:{strong.group(0)[:80]}")

    unit = MATH_UNIT_RE.search(problem)
    if unit and not reading_context and not has_options:
        reasons.append(f"quantitative_word_problem:{unit.group(0)[:80]}")

    en_word = EN_WORD_PROBLEM_RE.search(problem)
    if en_word and not reading_context and not has_options:
        reasons.append(f"english_word_problem:{en_word.group(0)[:80]}")

    zh_word = ZH_WORD_PROBLEM_RE.search(problem)
    if zh_word and not reading_context and not has_options:
        reasons.append(f"chinese_word_problem:{zh_word.group(0)[:80]}")

    # Catch short elementary word problems whose answer field is absent and
    # whose units (bananas, eggs, candies, allocation methods...) are open
    # ended rather than covered by the fixed unit list.
    if (
        len(problem) < 500
        and not reading_context
        and not has_options
        and len(NUM_RE.findall(problem)) >= 2
        and re.search(
            r"(?:多少|几种|几个|几次|几堆|几组|几排|几份|求|一共|总共|"
            r"还剩|分成|分给|每堆|每组|每排|每份|最后在哪)",
            problem,
        )
    ):
        reasons.append("short_quantitative_question")

    # Numeric-answer heuristic is only accepted when the prompt has multiple
    # numerical quantities, no multiple-choice wrapper, and no strong
    # reading/passage context. Numeric option labels such as "2" in science
    # multiple choice must not be mistaken for computed answers.
    numeric_answer = bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", answer))
    numerical_prompt = len(NUM_RE.findall(problem)) >= 2
    if numeric_answer and numerical_prompt and not reading_context and not has_options:
        reasons.append("numeric_answer+multiple_prompt_numbers")

    if reasons:
        return "math_high_confidence", reasons

    # Ambiguous math signals: keep for later classification rather than delete.
    ambiguous: list[str] = []
    if numeric_answer:
        if not has_options and not reading_context:
            return "math_high_confidence", ["numeric_answer_no_options"]
        ambiguous.append("numeric_answer_only")
    if re.search(r"(?:=|\+|\*|/|\^|\\frac|\\sqrt)", problem) and len(NUM_RE.findall(problem)) >= 1:
        ambiguous.append("operator_signal")
    if strong and has_options and not formal_notation:
        ambiguous.append("math_lexical_signal_in_choice")
    math_keyword = re.search(r"\bmath(?:ematics)?\b|数学|算术", text, re.I)
    if math_keyword and not has_options:
        # Explicitly labeled math tasks are high confidence even when answer
        # fields are absent.
        return "math_high_confidence", ["explicit_math_keyword"]
    if math_keyword and has_options:
        ambiguous.append("math_keyword_in_choice")
    if ambiguous:
        return "math_uncertain", ambiguous

    return "nonmath_candidate", []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, raw in cleaning.iter_snapshot(SOURCE):
        rows.append(
            {
                "uid": str(cleaning.pick(raw, "uid", "id", default=f"{path.stem}-{len(rows)}")),
                "dataset": "AquilaEdu",
                "instruction": str(cleaning.pick(raw, "instruction", "prompt", default="")),
                "input": str(cleaning.pick(raw, "input", "context", "question", default="")),
                "problem": str(
                    cleaning.pick(raw, "problem", "question", "input", "instruction", default="")
                ),
                "output": str(cleaning.pick(raw, "output", "response", "answer", default="")),
                "cot": str(cleaning.pick(raw, "output", "response", default="")),
                "quality": cleaning.pick(raw, "quality", "quality_score", "score", default=None),
                "domain": cleaning.pick(raw, "domain", "domains", "source", default=[]),
                "_source_file": path.name,
            }
        )
    return rows


def main() -> None:
    buckets: dict[str, list[dict[str, Any]]] = {
        "removed_unscored": [],
        "removed_low_quality": [],
        "removed_math_high_confidence": [],
        "math_uncertain": [],
        "kept_nonmath_candidate": [],
    }
    reason_counts: Counter[str] = Counter()
    scored_quality: list[float] = []
    total = 0

    for row in normalized_rows():
            total += 1
            quality = row.get("quality")
            routed = dict(row)
            routed["_filter_stage"] = "stage1"
            routed["_filter_date"] = DATE

            if not isinstance(quality, (int, float)):
                routed["_reason"] = "missing_quality_score"
                buckets["removed_unscored"].append(routed)
                continue

            scored_quality.append(float(quality))
            if quality < QUALITY_THRESHOLD:
                routed["_reason"] = "quality_below_threshold"
                routed["_quality_threshold"] = QUALITY_THRESHOLD
                buckets["removed_low_quality"].append(routed)
                continue

            label, reasons = classify_math(row)
            routed["_math_label"] = label
            routed["_math_reasons"] = reasons
            for reason in reasons:
                reason_counts[reason.split(":", 1)[0]] += 1

            if label == "math_high_confidence":
                routed["_reason"] = "math_redundant_with_specialist_corpora"
                buckets["removed_math_high_confidence"].append(routed)
            elif label == "math_uncertain":
                buckets["math_uncertain"].append(routed)
            else:
                buckets["kept_nonmath_candidate"].append(routed)

    for name, rows in buckets.items():
        write_jsonl(OUT / f"{name}.jsonl", rows)

    report = {
        "date": DATE,
        "input_stage1_kept": total,
        "quality_threshold": QUALITY_THRESHOLD,
        "removed_unscored": len(buckets["removed_unscored"]),
        "removed_low_quality": len(buckets["removed_low_quality"]),
        "quality_pass": (
            len(buckets["removed_math_high_confidence"])
            + len(buckets["math_uncertain"])
            + len(buckets["kept_nonmath_candidate"])
        ),
        "removed_math_high_confidence": len(buckets["removed_math_high_confidence"]),
        "math_uncertain": len(buckets["math_uncertain"]),
        "kept_nonmath_candidate": len(buckets["kept_nonmath_candidate"]),
        "remaining_for_next_stage": (
            len(buckets["math_uncertain"]) + len(buckets["kept_nonmath_candidate"])
        ),
        "math_reason_counts": dict(reason_counts),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
