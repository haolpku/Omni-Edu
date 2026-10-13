"""Non-math family cleaning: ARC, QASC, SciQ, SciInstruct (EN), CJEval, ScienceQA, Geometry3K."""

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
from common.text import is_empty as _is_empty, norm_key as norm

BASE = pools.source_dir("subject_competence")
HF = BASE / "huggingface"
GH = BASE / "github"


def s(x):
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def is_empty(x):
    return _is_empty(x, "{}")


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_pool(name, origin, kept_struct, removed_by_reason, routed_by_reason):
    out = pools.cleaned_dir("subject_competence", name)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "kept.jsonl", kept_struct)
    dump(out / "origin.jsonl", origin)
    for reason, rows in removed_by_reason.items():
        dump(out / f"removed_{reason}.jsonl", rows)
    for reason, rows in routed_by_reason.items():
        dump(out / f"routed_{reason}.jsonl", rows)
    rep = {
        "dataset": name,
        "origin_rows": len(origin),
        "kept_rows": len(kept_struct),
        "removed_by_reason": {k: len(v) for k, v in removed_by_reason.items()},
        "routed_by_reason": {k: len(v) for k, v in routed_by_reason.items()},
    }
    json.dump(rep, (out / "report.json").open("w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return rep


reports = {}


def build_mcq_options(choices):
    txt = list(choices.get("text", []))
    lab = list(choices.get("label", []))
    lines = [f"{l}. {t}" for l, t in zip(lab, txt)]
    return "\n".join(lines), dict(zip([str(l) for l in lab], txt)), [str(l) for l in lab]


# ---- ARC (Easy + Challenge) ----
def do_ARC():
    origin = []
    kept = []
    rem = {"empty": []}
    routed = {"answer_key_not_in_options": []}
    idx = 0
    for cfg in ["arc-easy", "arc-challenge"]:
        df = pd.read_parquet(f"{HF}/ai2_ARC/{cfg}/train-00000-of-00001.parquet")
        for _, r in df.iterrows():
            ch = r["choices"]
            ch = {"text": list(ch["text"]), "label": list(ch["label"])}
            optstr, lab2txt, labels = build_mcq_options(ch)
            ak = s(r["answerKey"]).strip()
            base = {
                "uid": f"ARC-{idx}",
                "dataset": "ARC",
                "problem": s(r["question"]),
                "options": optstr,
                "answer": ak,
                "answer_text": lab2txt.get(ak, ""),
                "cot": "",
                "meta": {
                    "config": cfg,
                    "src_id": s(r["id"]),
                    "lang": "en",
                    "subject": "science",
                    "no_cot": True,
                },
                "flags": [],
            }
            idx += 1
            origin.append(base)
            if is_empty(base["problem"]) or not labels:
                rr = dict(base)
                rr["_reason"] = "empty"
                rem["empty"].append(rr)
                continue
            if ak not in labels:
                rr = dict(base)
                rr["flags"] = ["answer_key_not_in_options"]
                routed["answer_key_not_in_options"].append(rr)
                continue
            kept.append(base)
    reports["ARC"] = write_pool(
        "ARC", origin, kept, rem, routed
    )


def do_QASC():
    df = pd.read_parquet(HF / "qasc" / "data" / "train-00000-of-00001.parquet")
    origin = []
    kept = []
    rem = {"empty": []}
    routed = {"answer_key_not_in_options": []}
    for i, r in df.iterrows():
        ch = r["choices"]
        ch = {"text": list(ch["text"]), "label": list(ch["label"])}
        optstr, lab2txt, labels = build_mcq_options(ch)
        ak = s(r["answerKey"]).strip()
        cf = s(r.get("combinedfact"))
        f1 = s(r.get("fact1"))
        f2 = s(r.get("fact2"))
        cot = cf if cf else (f"{f1} {f2}".strip())
        base = {
            "uid": f"QASC-{i}",
            "dataset": "QASC",
            "problem": s(r["question"]),
            "options": optstr,
            "answer": ak,
            "answer_text": lab2txt.get(ak, ""),
            "cot": cot,
            "meta": {
                "src_id": s(r["id"]),
                "lang": "en",
                "subject": "science",
                "cot_source": "combinedfact_or_facts",
                "fact1": f1,
                "fact2": f2,
            },
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or not labels:
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        if ak not in labels:
            rr = dict(base)
            rr["flags"] = ["answer_key_not_in_options"]
            routed["answer_key_not_in_options"].append(rr)
            continue
        kept.append(base)
    reports["QASC"] = write_pool(
        "QASC", origin, kept, rem, routed
    )


def do_SciQ():
    df = pd.read_parquet(HF / "sciq" / "data" / "train-00000-of-00001.parquet")
    origin = []
    kept = []
    rem = {"empty": []}
    routed = {}
    for i, r in df.iterrows():
        correct = s(r["correct_answer"])
        opts = [correct, s(r["distractor1"]), s(r["distractor2"]), s(r["distractor3"])]
        optstr = "\n".join(f"- {o}" for o in opts if o)
        support = s(r["support"])
        base = {
            "uid": f"SciQ-{i}",
            "dataset": "SciQ",
            "problem": s(r["question"]),
            "options": optstr,
            "answer": correct,
            "cot": support,
            "meta": {
                "lang": "en",
                "subject": "science",
                "has_support": bool(support.strip()),
                "distractors": [s(r["distractor1"]), s(r["distractor2"]), s(r["distractor3"])],
            },
            "flags": [] if support.strip() else ["no_support_cot"],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(correct):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        kept.append(base)
    reports["SciQ"] = write_pool("SciQ", origin, kept, rem, routed)


def do_sciinstruct_en():
    p = GH / "SciGLM" / "SciInstruct" / "SciInstruct" / "train_en_phy_chem.json"
    lines = [json.loads(l) for l in p.open() if l.strip()]
    origin = []
    kept = []
    rem = {"empty": []}
    routed = {}
    for i, r in enumerate(lines):
        summ = s(r.get("summary")).replace('\\"', '"')
        base = {
            "uid": f"sciinst-en-{i}",
            "dataset": "SciInstruct (EN)",
            "problem": s(r.get("content")),
            "answer": "",
            "cot": summ,
            "meta": {
                "lang": "en",
                "subject": "physics_chemistry",
                "note": "university_level_science",
                "answer_in_cot": True,
            },
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or is_empty(summ):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem["empty"].append(rr)
            continue
        kept.append(base)
    reports["SciInstruct (EN)"] = write_pool(
        "SciInstruct (EN)",
        origin,
        kept,
        rem,
        routed
    )


def cj_text(x):
    if isinstance(x, dict):
        return x.get("ques_content") and s(x.get("ques_content")) or s(x)
    return s(x)


CJ_OPTIONS_LABEL_RE = re.compile(r"(?:选项|选择|答案选项|options?)\s*[:：]", re.I)
CJ_OPTION_A_RE = re.compile(r"(?:^|[\s，。；;！!？?：:])(?:A|Ａ|A项|A选项)\s*[.．、)）]")
CJ_LABEL_RE = re.compile(
    r"^\s*(?:题目内容|问题内容|问题描述|材料内容|材料|题干|题目|问题|内容|question)\s*[:：]\s*",
    re.I,
)
CJ_BLANK_ONLY_RE = re.compile(r"^\s*(?:[（(]\s*[）)]\s*)+[。.!！?？]?\s*$")
CJ_QUESTION_HINT_RE = re.compile(
    r"[？?]|[（(]\s*[）)]|下列|以下|哪一|哪项|哪个|什么|为什么|怎么|如何|正确|错误|不正确"
)


def split_cjeval_content(text):
    text = s(text).strip()
    if not text:
        return "", "", ""
    label = CJ_OPTIONS_LABEL_RE.search(text)
    marker = CJ_OPTION_A_RE.search(text)
    if label and (marker is None or label.start() <= marker.start()):
        head, options = text[: label.start()], text[label.end() :]
    elif marker:
        head, options = text[: marker.start()], text[marker.start() :]
    else:
        head, options = text, ""
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])\s*", head) if part.strip()]
    question_index = None
    for index in range(len(sentences) - 1, -1, -1):
        sentence = sentences[index]
        if CJ_BLANK_ONLY_RE.match(sentence):
            question_index = max(0, index - 1)
            break
        if CJ_QUESTION_HINT_RE.search(sentence):
            question_index = index
            break
    if question_index is None:
        passage, problem = "", head.strip()
    else:
        passage = " ".join(sentences[:question_index])
        problem = " ".join(sentences[question_index:])
    return (
        CJ_LABEL_RE.sub("", passage).strip(),
        CJ_LABEL_RE.sub("", problem).strip() or head.strip(),
        options.strip(),
    )


def do_CJEval():
    files = sorted((GH / "CJEval" / "data" / "CJEval_data" / "train").glob("*.json"))
    CHOICE_TYPES = {
        "单选题",
        "多选题",
        "选择题",
        "单项选择",
        "词汇选择题",
        "阅读单选",
        "判断题",
        "阅读判断",
    }
    origin = []
    kept = []
    rem = {"empty_content_or_answer": []}
    routed = {}
    idx = 0
    for f in files:
        subj = f.split("train_")[-1].replace(".json", "")
        for line in f.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw_content = cj_text(r.get("ques_content"))
            passage, content, options = split_cjeval_content(raw_content)
            ans = r.get("ques_answer")
            ans_str = " | ".join(s(a) for a in ans) if isinstance(ans, list) else s(ans)
            analyze = r.get("ques_analyze")
            if isinstance(analyze, list):
                analyze = "\n".join(s(a) for a in analyze)
            elif isinstance(analyze, dict):
                analyze = s(analyze)
            else:
                analyze = s(analyze)
            qtype = s(r.get("ques_type"))
            base = {
                "uid": f"CJEval-{idx}",
                "dataset": "CJEval",
                "problem": content,
                "passage": passage,
                "options": options,
                "answer": ans_str,
                "cot": analyze,
                "meta": {
                    "subject": subj,
                    "ques_type": qtype,
                    "difficulty": s(r.get("ques_difficulty")),
                    "knowledges": r.get("ques_knowledges"),
                    "ques_content_raw": raw_content,
                    "lang": "zh",
                    "is_choice": qtype in CHOICE_TYPES,
                },
                "flags": [],
            }
            idx += 1
            origin.append(base)
            if is_empty(content) or is_empty(ans_str):
                rr = dict(base)
                rr["_reason"] = "empty_content_or_answer"
                rem["empty_content_or_answer"].append(rr)
                continue
            if is_empty(analyze):
                base["flags"].append("no_analysis_cot")
            kept.append(base)
    reports["CJEval"] = write_pool(
        "CJEval", origin, kept, rem, routed
    )


def do_scienceqa():
    df = pd.read_parquet(HF / "ScienceQA" / "data" / "train-00000-of-00001-1028f23e353fbe3e.parquet")
    origin = []
    text_kept = []
    routed_mm = []
    rem_empty = []
    for i, r in df.iterrows():
        img = r["image"]
        has_img = bool(img is not None and isinstance(img, dict) and img.get("bytes"))
        choices = list(r["choices"]) if r["choices"] is not None else []
        ans_idx = int(r["answer"]) if pd.notna(r["answer"]) else -1
        ans_text = choices[ans_idx] if 0 <= ans_idx < len(choices) else ""
        optstr = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(choices))
        sol = s(r["solution"])
        lec = s(r["lecture"])
        hint = s(r["hint"])
        cot = sol.strip()
        base = {
            "uid": f"scienceqa-{i}",
            "dataset": "ScienceQA (image)" if has_img else "ScienceQA (text)",
            "problem": s(r["question"]),
            "options": optstr,
            "answer": chr(65 + ans_idx) if ans_idx >= 0 else "",
            "answer_text": ans_text,
            "cot": cot,
            "passage": lec.strip(),
            "image_ref": {"origin_index": int(i), "has_image": has_img},
            "meta": {
                "has_image": has_img,
                "grade": s(r["grade"]),
                "subject": s(r["subject"]),
                "topic": s(r["topic"]),
                "skill": s(r["skill"]),
                "hint": hint,
                "task": s(r["task"]),
                "lang": "en",
            },
            "flags": [],
        }
        origin.append(base)
        if is_empty(base["problem"]) or ans_idx < 0 or not choices:
            rr = dict(base)
            rr["_reason"] = "empty_or_no_answer"
            rem_empty.append(rr)
            continue
        if has_img:
            rr = dict(base)
            rr["flags"] = ["multimodal_deferred"]
            routed_mm.append(rr)
        else:
            text_kept.append(base)
    reports["ScienceQA (text)"] = write_pool(
        "ScienceQA (text)",
        origin,
        text_kept,
        {"empty_or_no_answer": rem_empty},
        {},
    )
    reports["ScienceQA (image)"] = write_pool(
        "ScienceQA (image)",
        origin,
        routed_mm,
        {},
        {"multimodal_deferred": routed_mm},
    )


def do_Geometry3K():
    df = pd.read_parquet(HF / "geometry3k" / "data" / "train-00000-of-00001.parquet")
    origin = []
    routed_mm = []
    rem_empty = []
    for i, r in df.iterrows():
        imgs = r["images"]
        has_img = bool(imgs is not None and len(imgs) > 0)
        prob = s(r["problem"])
        ans = s(r["answer"])
        base = {
            "uid": f"Geometry3K-{i}",
            "dataset": "Geometry3K",
            "problem": prob,
            "answer": ans,
            "cot": "",
            "image_ref": {
                "origin_index": int(i),
                "has_image": has_img,
                "n_images": int(len(imgs)) if has_img else 0,
            },
            "meta": {
                "has_image": has_img,
                "subject": "math_geometry",
                "lang": "en",
                "no_cot": True,
            },
            "flags": [],
        }
        origin.append(base)
        if is_empty(prob) or is_empty(ans):
            rr = dict(base)
            rr["_reason"] = "empty"
            rem_empty.append(rr)
            continue
        rr = dict(base)
        rr["flags"] = ["multimodal_deferred"]
        routed_mm.append(rr)
    reports["Geometry3K"] = write_pool(
        "Geometry3K",
        origin,
        list(routed_mm),
        {"empty": rem_empty},
        {"multimodal_deferred": routed_mm},
    )


do_ARC()
print("ARC done")
do_QASC()
print("QASC done")
do_SciQ()
print("SciQ done")
do_sciinstruct_en()
print("sciinstruct-en done")
do_CJEval()
print("CJEval done")
do_scienceqa()
print("scienceqa done")
do_Geometry3K()
print("Geometry3K done")

print("\nreports:")
print(json.dumps(reports, ensure_ascii=False, indent=2))
