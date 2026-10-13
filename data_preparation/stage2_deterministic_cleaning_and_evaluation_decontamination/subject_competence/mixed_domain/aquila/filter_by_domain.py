#!/usr/bin/env python3
"""Final rule routing for deduplicated Aquila candidates.

The source has no domain labels. This pass therefore uses conservative,
structure-aware rules:
1. Preserve genuine reading comprehension (substantial passage before the
   final question), regardless of passage topic.
2. Remove remaining math-uncertain and obvious quantitative problems.
3. Remove clear professional/generalist domains (code, business, clinical
   medicine, legal advice, lifestyle/recommendation, marketing).
4. Keep school subjects, basic science, humanities and commonsense QA.
5. Route genuinely ambiguous short instructions to an uncertain bucket.
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
from common import pools

from typing import Any

PIPE = pools.STAGE2 / "subject_competence"
SRC = PIPE / "AquilaEdu/near_dedup/kept.jsonl"
OUT = PIPE / "AquilaEdu/cleaned"
DATE = "2026-08-05"

QUESTION_MARK_RE = re.compile(r"(?is)(?:question|问题)\s*[:：]")
OPTION_RE = re.compile(
    r"(?is)(?:options?|选项|选择)\s*[:：]|(?:^|[\s　])(?:\([A-H]\)|[A-H][.、．])"
)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def true_reading(problem: str) -> bool:
    matches = list(QUESTION_MARK_RE.finditer(problem))
    if matches:
        # Use the final question marker; substantial preceding context means
        # this is a passage task rather than "Question: standalone QA".
        prefix = problem[: matches[-1].start()]
        suffix = problem[matches[-1].end() :]
        return len(prefix.strip()) >= 260 and len(suffix.strip()) >= 20
    # Chinese datasets sometimes place the question without an explicit marker.
    return (
        len(problem) >= 900
        and bool(OPTION_RE.search(problem))
        and bool(re.search(r"阅读|文章|材料|短文|作者|段落|passage|article", problem, re.I))
    )


CODE_RE = re.compile(
    r"(?:编写|实现|调试|运行|解释).{0,16}(?:代码|程序|函数|脚本|算法)|"
    r"\b(?:python|javascript|java|c\+\+|c#|html|css|sql|react|vue|"
    r"node\.?js|php|docker|linux|regex|database|programming|"
    r"software development)\b",
    re.I,
)
BUSINESS_RE = re.compile(
    r"(?:客户忠诚|市场营销|营销策略|销售策略|商业计划|投资组合|"
    r"财务报表|企业管理|客户服务|产品推广|品牌推广|广告文案|"
    r"供应链管理|绩效管理)|"
    r"\b(?:customer retention|marketing strategy|business plan|"
    r"investment portfolio|financial statement|sales strategy|"
    r"customer service|product marketing)\b",
    re.I,
)
MEDICAL_RE = re.compile(
    r"(?:请|如何|给出|制定).{0,12}(?:诊断|治疗方案|用药建议|处方|剂量)|"
    r"(?:患者|病人).{0,20}(?:诊断|用药|治疗)|"
    r"\b(?:diagnose|treatment plan|prescription|dosage|medical advice)\b",
    re.I,
)
LEGAL_RE = re.compile(
    r"(?:法律意见|法律咨询|诉讼策略|合同起草|律师函)|"
    r"\b(?:legal advice|lawsuit strategy|draft a contract|legal memorandum)\b",
    re.I,
)
LIFESTYLE_RE = re.compile(
    r"(?:旅游攻略|旅行计划|餐厅推荐|电影推荐|电视剧推荐|购物推荐|"
    r"穿搭建议|减肥计划|健身计划|食谱推荐|求职信|简历优化|"
    r"社交媒体文案|产品评论)|"
    r"\b(?:travel itinerary|restaurant recommendation|movie recommendation|"
    r"shopping recommendation|fitness plan|weight loss plan|"
    r"cover letter|resume optimization|social media post|product review)\b",
    re.I,
)
GENERIC_RE = re.compile(
    r"^(?:列出|提出|生成|撰写|写一篇|制定|创建|设计)"
    r".{0,50}(?:方法|策略|计划|方案|文案|邮件|故事|笑话|产品|服务)|"
    r"^(?:list|generate|write|create|design|propose)"
    r".{0,60}(?:methods|strategies|plan|email|story|joke|product|service)",
    re.I | re.S,
)

SCHOOL_RE = re.compile(
    r"(?:中学|小学|高中|初中|学生|课堂|课程|考试|题目|选择题|"
    r"语文|英语|历史|地理|政治|物理|化学|生物|科学|文学|"
    r"文言文|古诗|诗歌|哲学|文化|教育|school|student|teacher|"
    r"history|geography|physics|chemistry|biology|science|literature)",
    re.I,
)
QUANT_RE = re.compile(
    r"(?:多少|几种|几个|几次|几堆|几组|几排|百分之几|求.{0,10}值|"
    r"一共|总共|还剩|平均每|速度|路程|面积|周长|体积)|"
    r"\b(?:how many|how much|calculate|compute|solve|equation|"
    r"probability|perimeter|volume|area)\b",
    re.I,
)


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    instruction = str(row.get("instruction") or "")
    problem = str(row.get("input") or row.get("problem") or instruction)
    text = f"{instruction}\n{problem}".strip()

    if true_reading(problem):
        return "keep_relevant", ["reading_comprehension"]

    # Rows entering from math_uncertain remain marked. Once genuine reading is
    # excluded, this bucket is overwhelmingly math and can be safely dropped.
    if row.get("_math_label") == "math_uncertain":
        return "remove_math", ["prior_math_uncertain_nonreading"]

    if (
        len(problem) < 700
        and len(NUM_RE.findall(problem)) >= 2
        and QUANT_RE.search(problem)
        and not OPTION_RE.search(problem)
    ):
        return "remove_math", ["quantitative_problem"]

    # After mathematical candidates have been routed, standalone multiple-
    # choice questions are educational/knowledge QA. Domain words such as
    # "react", "program", "patient", or "marketing" should not cause them to
    # be mistaken for professional code/medical/business instructions.
    if OPTION_RE.search(problem):
        return "keep_relevant", ["multiple_choice_knowledge"]

    irrelevant: list[str] = []
    for name, pattern in [
        ("code_software", CODE_RE),
        ("business_finance", BUSINESS_RE),
        ("medical_clinical", MEDICAL_RE),
        ("legal_professional", LEGAL_RE),
        ("lifestyle_recommendation", LIFESTYLE_RE),
    ]:
        if pattern.search(text):
            irrelevant.append(name)
    if irrelevant:
        # A substantial multiple-choice passage may merely discuss medicine,
        # business, or software as its subject matter. Keep it as reading
        # comprehension instead of treating the passage topic as the task.
        if len(problem) >= 450 and OPTION_RE.search(problem):
            return "keep_relevant", ["reading_comprehension_domain_text"]
        return "remove_irrelevant", irrelevant
    if GENERIC_RE.search(text) and not SCHOOL_RE.search(text):
        return "remove_irrelevant", ["generic_instruction_task"]

    # Short open prompts without QA structure or a school-domain signal remain
    # uncertain rather than being silently accepted.
    if (
        len(problem) < 160
        and not row.get("answer")
        and not OPTION_RE.search(problem)
        and not SCHOOL_RE.search(text)
    ):
        return "uncertain_relevance", ["short_unlabeled_instruction"]

    return "keep_relevant", ["school_or_knowledge_candidate"]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    buckets = {
        "kept_relevant": [],
        "removed_irrelevant": [],
        "removed_math": [],
        "uncertain_relevance": [],
    }
    reasons: Counter[str] = Counter()
    total = 0
    with SRC.open(encoding="utf-8") as src:
        for line in src:
            row = json.loads(line)
            total += 1
            label, why = classify(row)
            routed = dict(row)
            routed["_relevance_label"] = label
            routed["_relevance_reasons"] = why
            routed["_relevance_stage"] = "stage1"
            routed["_relevance_date"] = DATE
            for reason in why:
                reasons[reason] += 1
            if label == "remove_irrelevant":
                routed["_reason"] = "irrelevant_to_k12_subject_competence"
                buckets["removed_irrelevant"].append(routed)
            elif label == "remove_math":
                routed["_reason"] = "math_redundant_with_specialist_corpora"
                buckets["removed_math"].append(routed)
            elif label == "uncertain_relevance":
                buckets["uncertain_relevance"].append(routed)
            else:
                buckets["kept_relevant"].append(routed)

    for name, rows in buckets.items():
        target = OUT / ("kept.jsonl" if name == "kept_relevant" else f"{name}.jsonl")
        write(target, rows)
    report = {
        "date": DATE,
        "input": total,
        **{name: len(rows) for name, rows in buckets.items()},
        "reason_counts": dict(reasons),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
