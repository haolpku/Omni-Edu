
from pathlib import Path

import json
import re

import pandas as pd

import sys as _sys

_COMMON_ROOT = Path(__file__).resolve()
while _COMMON_ROOT.name != "data_preparation":
    _COMMON_ROOT = _COMMON_ROOT.parent
if str(_COMMON_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_COMMON_ROOT))
from common import pools
from common.text import is_empty, norm_key as norm

BASE = pools.source_dir("subject_competence")
HF = BASE / "huggingface"
GH = BASE / "github"


def s(x):
    return "" if x is None else str(x)


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_boxed(sol):
    res = None
    i = 0
    while True:
        j = sol.find(r"\boxed{", i)
        if j < 0:
            break
        k = j + len(r"\boxed{")
        depth = 1
        start = k
        while k < len(sol) and depth > 0:
            if sol[k] == "{":
                depth += 1
            elif sol[k] == "}":
                depth -= 1
            k += 1
        if depth == 0:
            res = sol[start : k - 1].strip()
        i = k
    return res or ""


def write_pool(name, origin, kept_struct, removed_by_reason):
    out = pools.cleaned_dir("subject_competence", name)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "kept.jsonl", kept_struct)
    dump(out / "origin.jsonl", origin)
    for reason, rows in removed_by_reason.items():
        dump(out / f"removed_{reason}.jsonl", rows)
    rep = {
        "dataset": name,
        "origin_rows": len(origin),
        "kept_rows": len(kept_struct),
        "removed_by_reason": {k: len(v) for k, v in removed_by_reason.items()},
        "removed_rows": sum(len(v) for v in removed_by_reason.values()),
    }
    json.dump(rep, (out / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rep


reports = {}


# ================= OpenR1 (default) =================
def do_openr1():
    files = sorted((HF / "OpenR1-Math-220k" / "data").glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    origin = []
    kept = []
    rem = {"all_generations_wrong": [], "empty_problem_or_answer": []}
    for i, r in df.iterrows():
        gens = list(r["generations"]) if r["generations"] is not None else []
        cmv = list(r["correctness_math_verify"]) if r["correctness_math_verify"] is not None else []
        cc = int(r["correctness_count"]) if pd.notna(r["correctness_count"]) else 0
        base = {
            "uid": f"openr1-{s(r['uuid'])[:12]}-{i}",
            "dataset": "OpenR1",
            "problem": s(r["problem"]),
            "answer": s(r["answer"]),
            "cot": s(r["solution"]),
            "meta": {
                "source": s(r["source"]),
                "problem_type": s(r["problem_type"]),
                "correctness_count": cc,
                "n_generations": len(gens),
            },
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(base["answer"]):
            rr = dict(base)
            rr["_reason"] = "empty_problem_or_answer"
            rem["empty_problem_or_answer"].append(rr)
            continue
        idx = next((j for j, ok in enumerate(cmv) if ok), None)
        if idx is None:
            rr = dict(base)
            rr["_reason"] = "all_generations_wrong"
            rem["all_generations_wrong"].append(rr)
            continue
        rec = dict(base)
        rec["cot"] = s(gens[idx])
        rec["flags"] = ["cot=verified_generation"]
        rec["meta"] = dict(base["meta"])
        rec["meta"]["olympiad_level"] = True
        kept.append(rec)
    reports["OpenR1"] = write_pool(
        "OpenR1", origin, kept, rem
    )


# ================= gsm8k (main train) =================
def do_gsm8k():
    df = pd.read_parquet(HF / "gsm8k" / "main" / "train-00000-of-00001.parquet")
    origin = []
    kept = []
    rem = {"no_final_answer": [], "empty": []}
    for i, r in df.iterrows():
        ans_full = s(r["answer"])
        m = re.search(r"####\s*(.+)\s*$", ans_full)
        final = m.group(1).strip() if m else ""
        base = {
            "uid": f"gsm8k-{i}",
            "dataset": "GSM8K",
            "problem": s(r["question"]),
            "answer": final,
            "cot": ans_full,
            "meta": {"lang": "en"},
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(ans_full):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        if not final:
            rr = dict(base)
            rr["_reason"] = "no_final_answer"
            rem["no_final_answer"].append(rr)
            continue
        kept.append(base)
    reports["GSM8K"] = write_pool(
        "GSM8K", origin, kept, rem
    )


# ================= MATH (hendrycks, train only) =================
def do_math():
    files = [
        f for f in (HF / "hendrycks_math").rglob("*.parquet") if "train" in str(f)
    ]
    origin = []
    kept = []
    rem = {"no_boxed_answer": [], "empty": []}
    idx = 0
    for f in sorted(files):
        df = pd.read_parquet(f)
        for _, r in df.iterrows():
            sol = s(r["solution"])
            ans = extract_boxed(sol)
            base = {
                "uid": f"math-{idx}",
                "dataset": "MATH",
                "problem": s(r["problem"]),
                "answer": ans,
                "cot": sol,
                "meta": {"level": s(r["level"]), "type": s(r["type"])},
                "flags": [],
            }
            idx += 1
            origin.append(base)
            if is_empty(base["problem"]) or is_empty(sol):
                rr = dict(base)
                rr["_reason"] = "empty"
                rem["empty"].append(rr)
                continue
            if not ans:
                rr = dict(base)
                rr["_reason"] = "no_boxed_answer"
                rem["no_boxed_answer"].append(rr)
                continue
            kept.append(base)
    reports["MATH"] = write_pool(
        "MATH", origin, kept, rem
    )


# ================= math_qa (train) =================
def do_MathQA():
    d = json.load((HF / "math_qa" / "raw" / "train.json").open())
    origin = []
    kept = []
    rem = {"bad_correct_letter": [], "empty": []}
    for i, r in enumerate(d):
        correct = s(r.get("correct")).strip().lower()
        base = {
            "uid": f"MathQA-{i}",
            "dataset": "MathQA",
            "problem": s(r.get("Problem")),
            "answer": correct,
            "cot": s(r.get("Rationale")),
            "options": s(r.get("options")),
            "meta": {"annotated_formula": s(r.get("annotated_formula")), "lang": "en"},
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(base["cot"]) or is_empty(base["options"]):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        if correct not in ("a", "b", "c", "d", "e"):
            rr = dict(base)
            rr["_reason"] = "bad_correct_letter"
            rem["bad_correct_letter"].append(rr)
            continue
        kept.append(base)
    reports["MathQA"] = write_pool("MathQA", origin, kept, rem)


# ================= SciInstruct cn_math =================
def do_sciinstruct_cn():
    p = GH / "SciGLM" / "SciInstruct" / "SciInstruct" / "train_cn_math.json"
    lines = [json.loads(l) for l in p.open() if l.strip()]
    origin = []
    kept = []
    rem = {"empty": []}
    ANS = re.compile(r"【答案】\s*(.+?)(?:【解析】|$)", re.S)
    for i, r in enumerate(lines):
        summ = s(r.get("summary"))
        m = ANS.search(summ)
        ans = m.group(1).strip()[:200] if m else ""
        base = {
            "uid": f"sciinst-cn-{i}",
            "dataset": "SciInstruct (CN math)",
            "problem": s(r.get("content")),
            "answer": ans,
            "cot": summ,
            "meta": {"lang": "zh", "note": "university_level_science"},
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(summ):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        kept.append(base)
    reports["SciInstruct (CN math)"] = write_pool(
        "SciInstruct (CN math)", origin, kept, rem
    )


do_openr1()
print("openr1 done")
do_gsm8k()
print("gsm8k done")
do_math()
print("math done")
do_MathQA()
print("MathQA done")
do_sciinstruct_cn()
print("sciinstruct done")

print("\nreports:")
print(json.dumps(reports, ensure_ascii=False, indent=2))
