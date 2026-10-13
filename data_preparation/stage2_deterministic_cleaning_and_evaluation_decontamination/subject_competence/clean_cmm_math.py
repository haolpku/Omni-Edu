
from pathlib import Path

import json
import os
import re
import shutil

import pandas as pd

import sys as _sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools
from common.text import is_empty, norm_key as norm

SRC = pools.source_dir("subject_competence") / "huggingface" / "cmm-math" / "train_data.parquet"
OUT = pools.cleaned_dir("subject_competence", "cmm-math")
OUT.mkdir(parents=True, exist_ok=True)


def s(x):
    return "" if x is None else str(x)


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


OPT = re.compile(r"(?:^|[^A-Za-z])([A-Ha-h])\s*[\.．、]")


def opt_letters(o):
    return {m.upper() for m in OPT.findall(s(o))}


ANS_REC = re.compile(r"答案\s*[:：]?\s*(.+?)(?:\n|解析|$)")

df = pd.read_parquet(SRC)
N0 = len(df)

origin = []
for _, r in df.iterrows():
    origin.append(
        {
            "uid": f"cmm-math-{r['id']}",
            "dataset": None,
            "problem": s(r["question"]),
            "answer": s(r["answer"]),
            "cot": s(r["analysis"]),
            "options": "" if is_empty(r["options"]) else s(r["options"]),
            "image_ref": [] if is_empty(r["image"]) else json.loads(s(r["image"])),
            "meta": {"level": s(r["level"]), "subject": s(r["subject"])},
            "flags": [],
        }
    )
dump(OUT / "origin.jsonl", origin)

L1 = {
    k: []
    for k in [
        "kept",
        "removed_empty_question",
        "removed_empty_answer",
        "routed_multimodal",
        "routed_answer_only_valid",
        "removed_big_question_no_cot",
    ]
}
recovered = 0


def has_img(rec):
    return bool(rec["image_ref"]) or any(
        "<ImageHere>" in rec[k] for k in ("problem", "answer", "cot")
    )


def answer_only_subtype(rec):
    q, a = rec["problem"], rec["answer"].strip()
    if rec["options"]:
        return "choice_options_col"
    if re.search(r"(?m)^\s*[A-D][\.、]", q) or re.search(r"[A-D][\.、].{0,40}[A-D][\.、]", q):
        return "choice_inline"
    if re.search(r"[（(]\s*[)）]", q):
        return "fill_or_choice_slot"
    if a in ("对", "错", "正确", "错误", "√", "×") or "√" in a or "×" in a or "times" in a:
        return "judge"
    if len(a) <= 6:
        return "short_fill"
    return "big_question_solution_missing_or_in_answer"


ao_subtypes = {}
for rec in origin:
    r = dict(rec)  # copy
    if is_empty(r["problem"]):
        r["_reason"] = "empty_question"
        L1["removed_empty_question"].append(r)
        continue
    if is_empty(r["answer"]):
        m = ANS_REC.search(r["cot"])
        if m and m.group(1).strip():
            r["answer"] = m.group(1).strip()
            r["flags"] = r["flags"] + ["answer_recovered"]
            recovered += 1
        else:
            r["_reason"] = "empty_answer_unrecoverable"
            L1["removed_empty_answer"].append(r)
            continue
    if has_img(r):
        r["_route"] = "multimodal"
        L1["routed_multimodal"].append(r)
        continue
    if is_empty(r["cot"]):
        st = answer_only_subtype(r)
        ao_subtypes[st] = ao_subtypes.get(st, 0) + 1
        if st == "big_question_solution_missing_or_in_answer":
            r["_reason"] = "big_question_no_cot"
            r["_subtype"] = st
            L1["removed_big_question_no_cot"].append(r)
            continue
        r["_route"] = "answer_only_valid"
        r["_subtype"] = st
        L1["routed_answer_only_valid"].append(r)
        continue
    L = opt_letters(r["options"])
    a = r["answer"].strip()
    if L and re.fullmatch(r"[A-H]", a) and a not in L:
        r["flags"] = r["flags"] + ["mcq_answer_mismatch_unreliable"]
    L1["kept"].append(r)

for k, v in L1.items():
    if k == "kept":
        continue
    if k not in ("kept", "routed_multimodal", "routed_answer_only_valid"):
        dump(OUT / f"{k}.jsonl", v)

seen = {}
kept2 = []
dups = []
for r in L1["kept"]:
    key = norm(r["problem"] + "||" + r["options"])
    if key in seen:
        rr = dict(r)
        rr["_reason"] = "exact_duplicate"
        rr["_kept_uid"] = seen[key]
        dups.append(rr)
        continue
    seen[key] = r["uid"]
    kept2.append(r)
dump(OUT / "removed_duplicates.jsonl", dups)


def dump_pool(name, rows):
    for r in rows:
        r["dataset"] = name
    out = pools.cleaned_dir("subject_competence", name)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "kept.jsonl", rows)
    return len(rows)


dump_pool("CMM-Math (text cot)", kept2)
dump_pool("CMM-Math (text answer)", L1["routed_answer_only_valid"])
dump_pool("CMM-Math (image)", L1["routed_multimodal"])

hit = den = 0
for r in L1["kept"]:
    a = r["answer"].strip()
    if a and len(a) <= 20:
        den += 1
        if a in r["cot"] or norm(a) in norm(r["cot"]):
            hit += 1
mcq_flag = sum(1 for r in L1["kept"] if "mcq_answer_mismatch_unreliable" in r["flags"])

report = {
    "dataset": ["CMM-Math (text cot)", "CMM-Math (text answer)", "CMM-Math (image)"],
    "source": SRC,
    "origin_rows": N0,
    "structural_cleaning": {
        "kept_text_with_cot": len(L1["kept"]),
        "removed_empty_question": len(L1["removed_empty_question"]),
        "removed_empty_answer_unrecoverable": len(L1["removed_empty_answer"]),
        "answer_recovered_from_analysis": recovered,
        "routed_multimodal": len(L1["routed_multimodal"]),
        "routed_answer_only_valid": len(L1["routed_answer_only_valid"]),
        "removed_big_question_no_cot": len(L1["removed_big_question_no_cot"]),
        "answer_only_subtypes_seen": ao_subtypes,
    },
    "dedup": {"kept_rows": len(kept2), "removed_duplicates": len(dups)},
    "flags": {"mcq_answer_mismatch_unreliable": mcq_flag},
    "soft_signal_answer_in_analysis_pct": round(hit / den * 100, 2) if den else None,
    "soft_signal_denom": den,
    "hard_removed_total": len(L1["removed_empty_question"])
    + len(L1["removed_empty_answer"])
    + len(L1["removed_big_question_no_cot"])
    + len(dups),
}
json.dump(report, (OUT / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
