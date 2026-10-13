#!/usr/bin/env python3
"""Route COIG-CQIA Stage-1 kept rows by subject-competence relevance.

Conservative K12-oriented policy:
- Keep explicit middle-school exam data and curriculum-relevant Chinese culture.
- Keep logic reasoning and selected basic encyclopedia/science knowledge.
- Drop social/general instruction, medical, finance, law/postgraduate, media,
  programming/NLP annotation, lifestyle and other clearly unrelated sources.
- Keep outputs no-delete: all excluded rows are written with a reason.
"""

from __future__ import annotations

import json
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
OUT = PIPE / "COIG-CQIA/relevance"
SOURCE = pools.source_dir("subject_competence") / "huggingface" / "COIG-CQIA"
DATE = "2026-08-05"


ALWAYS_KEEP_FILES = {
    "exam/coig_exam_sampled_clean.jsonl",
    "logi_qa/logi-qa.jsonl",
    "chinese_traditional/chengyu.jsonl",
    "chinese_traditional/poem.jsonl",
    "chinese_traditional/trad-multi-choice-100.jsonl",
    "chinese_traditional/trad-multi-choice-100-2.jsonl",
    "chinese_traditional/trad-multi-choice-40.jsonl",
    "chinese_traditional/translate_classical_chinese.jsonl",
    "wiki/10why_final.jsonl",
}

ALWAYS_DROP_PREFIXES = (
    "finance/",
    "douban/",
    "zhihu/",
    "xhs/",
    "wikihow/",
    "segmentfault/",
    "human_value/",
    "ruozhiba/",
)

ALWAYS_DROP_FILES = {
    "exam/law_gee_exam_clean.jsonl",
    "exam/kaoyan.jsonl",
    "wiki/bkmy_medicine.jsonl",
    "wiki/bkmy_symptom.jsonl",
    "wiki/baobao.jsonl",
    "wiki/51zyzy.jsonl",
    "wiki/agriculture.jsonl",
    "coig_pc/coig_pc_core_sample.jsonl",
}

# zgbk entries are retained only when at least one explicit domain tag is
ZGBK_KEEP_TAGS = {
    "物理学",
    "化学",
    "生物学",
    "微生物学",
    "天文学",
    "基本天文学",
    "天气学",
    "大气科学",
    "地理学",
    "中国地理",
    "中国城市历史地理变迁",
    "历史学",
    "综合史",
    "科学技术史",
    "文学",
    "中国文学",
    "中国少数民族文学",
    "少数民族文学",
    "语言文字",
    "世界诸语言",
    "汉藏语系",
    "哲学",
    "考古学",
    "文物",
    "生态学",
    "景观生态学",
    "区域生态学",
    "矿物学",
    "地球物理学",
    "海洋学",
    "海洋",
    "数学",
    "力学",
}

# digital_wiki is mostly basic electronics. Keep it as optional K12 science /
# information-technology knowledge rather than dropping it with programming.
OPTIONAL_KEEP_FILES = {"wiki/digital_wiki.jsonl"}


def route(row: dict[str, Any]) -> tuple[bool, str]:
    source = str(row.get("_source_file") or "")
    if source in ALWAYS_KEEP_FILES:
        return True, "explicit_k12_or_subject_source"
    if source.startswith(ALWAYS_DROP_PREFIXES):
        return False, "unrelated_source_family"
    if source in ALWAYS_DROP_FILES:
        return False, "unrelated_or_non_k12_source"
    if source in OPTIONAL_KEEP_FILES:
        return True, "basic_science_information_technology"
    if source == "wiki/zgbk.jsonl":
        domains = set(map(str, row.get("domain") or []))
        if domains & ZGBK_KEEP_TAGS:
            return True, "selected_curriculum_encyclopedia_domain"
        return False, "encyclopedia_domain_not_k12_priority"
    return False, "unclassified_not_k12_priority"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = SOURCE
    for path, raw in cleaning.iter_snapshot(root):
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = path.name
        rows.append(
            {
                "uid": str(cleaning.pick(raw, "uid", "id", default=f"{path.stem}-{len(rows)}")),
                "dataset": "COIG-CQIA",
                "instruction": str(cleaning.pick(raw, "instruction", "prompt", default="")),
                "input": str(cleaning.pick(raw, "input", "context", default="")),
                "problem": str(
                    cleaning.pick(raw, "problem", "question", "instruction", default="")
                ),
                "output": str(cleaning.pick(raw, "output", "response", "answer", default="")),
                "cot": str(cleaning.pick(raw, "output", "response", default="")),
                "domain": cleaning.pick(raw, "domain", "domains", default=[]),
                "_source_file": relative,
            }
        )
    return rows


def main() -> None:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    kept_files: Counter[str] = Counter()
    removed_files: Counter[str] = Counter()
    kept_domains: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for row in normalized_rows():
            ok, reason = route(row)
            routed = dict(row)
            routed["_relevance_stage"] = "stage1"
            routed["_relevance_date"] = DATE
            routed["_relevance_reason"] = reason
            source = str(row.get("_source_file") or "")
            reasons[reason] += 1
            if ok:
                kept.append(routed)
                kept_files[source] += 1
                for domain in row.get("domain") or []:
                    kept_domains[str(domain)] += 1
            else:
                routed["_reason"] = "irrelevant_to_k12_subject_competence"
                removed.append(routed)
                removed_files[source] += 1

    write_jsonl(OUT / "kept_relevant.jsonl", kept)
    write_jsonl(OUT / "removed_irrelevant.jsonl", removed)
    report = {
        "date": DATE,
        "input": len(kept) + len(removed),
        "kept_relevant": len(kept),
        "removed_irrelevant": len(removed),
        "kept_ratio": len(kept) / max(len(kept) + len(removed), 1),
        "policy": "conservative K12 subject competence",
        "kept_by_source": dict(kept_files.most_common()),
        "removed_by_source": dict(removed_files.most_common()),
        "kept_domain_top": dict(kept_domains.most_common(50)),
        "route_reasons": dict(reasons),
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
