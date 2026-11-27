#!/usr/bin/env python3
"""Rebuild the static Pro comparison and downloadable scores from raw metrics."""

import csv
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = (
    ("curriculum", "Curriculum grounding", "Connect questions to the curriculum."),
    ("solving", "K–12 problem solving", "Solve school problems across subjects."),
    ("tutoring", "Pedagogical tutoring", "Diagnose gaps and guide the next step."),
)
FORMULAS = (
    "[(K12-Bench EM + K12-Bench F1) / 2 + MathFish Acc. + EDUMATH MaC] / 3",
    "(GAOKAO-Bench Full + EXAMS-V + MDK12-Bench Full) / 3",
    "[MathTutor Overall + TutorBench + (LongTutor-E + 25 × (LongTutor-T − 1)) / 2] / 3",
)


def scores(metrics):
    v = {k: Decimal(value) for k, value in metrics.items()}
    for key, value in v.items():
        low, high = (1, 5) if key == "longtutor_t" else (0, 100)
        if not low <= value <= high:
            raise ValueError(f"Invalid {key}: {value}")
    return (
        ((v["k12_bench_em"] + v["k12_bench_f1"]) / 2 + v["mathfish_acc"] + v["edumath_mac"]) / 3,
        (v["gaokao_bench_full"] + v["exams_v"] + v["mdk12_bench_full"]) / 3,
        (v["mathtutor_overall"] + v["tutorbench"] + (v["longtutor_e"] + 25 * (v["longtutor_t"] - 1)) / 2) / 3,
    )


def fmt(value):
    return str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def render(models):
    featured = [m for m in models if m["kind"] in ("pro", "frontier")]
    pro27 = next(m for m in models if m["name"] == "OmniEdu-Pro-27B")
    frontier = [m for m in featured if m["kind"] == "frontier"]
    output = ['<div class="score-charts">']
    for i, (key, title, description) in enumerate(CATEGORIES):
        best = max(frontier, key=lambda m: m["scores"][i])
        output.append(f'<article class="score-chart" aria-labelledby="chart-{key}"><h3 id="chart-{key}">{title}</h3><p>{description}</p>')
        output.append('<ol class="score-list">')
        for m in sorted(featured, key=lambda m: m["scores"][i], reverse=True):
            value = fmt(m["scores"][i])
            name = html.escape(m["name"])
            output.append(f'<li class="score-row {m["kind"]}"><div class="score-label"><span>{name}</span><strong>{value}</strong></div><div class="score-track" aria-hidden="true"><div class="score-fill" style="width:{value}%"></div></div></li>')
        output.append('</ol><div class="score-axis" aria-hidden="true"><span>0</span><span>50</span><span>100</span></div>')
        ours, theirs = fmt(pro27["scores"][i]), fmt(best["scores"][i])
        # Compare the displayed scores so the annotation matches the visible bars.
        difference = Decimal(ours) - Decimal(theirs)
        delta = f'{"+" if difference >= 0 else "−"}{fmt(abs(difference))}'
        output.append(f'<p class="score-comparison"><b>27B-Pro {ours}</b> · {html.escape(best["name"])} {theirs}<br><span>{delta} points vs. the highest frontier score in this category</span></p></article>')
    output.append('</div>')
    output.append('<details class="benchmark"><summary>Full comparison<span>13 models · scores out of 100</span></summary><div class="table-scroll" role="region" aria-label="OmniEdu-Pro full composite scores" tabindex="0"><table id="pro-score-table"><caption>September 22, 2026 · MathTutorBench included</caption><thead><tr><th scope="col">Model</th>')
    output.extend(f'<th scope="col">{title}</th>' for _, title, _ in CATEGORIES)
    output.append('</tr></thead><tbody>')
    for m in models:
        style = ' class="ours"' if m["kind"] == "pro" else ''
        output.append(f'<tr{style}><th scope="row">{html.escape(m["name"])}</th>' + ''.join(f'<td>{fmt(s)}</td>' for s in m["scores"]) + '</tr>')
    output.append('</tbody></table></div></details>')
    output.append('<details class="benchmark scoring" id="scoring"><summary>How the three scores are calculated<span>Benchmark-level equal weights</span></summary><div class="formula-list">')
    for (_, title, _), formula in zip(CATEGORIES, FORMULAS):
        output.append(f'<div><h4>{title}</h4><code>{html.escape(formula)}</code></div>')
    output.append('<p>Each category averages three benchmarks equally. K12-Bench EM and F1 share one benchmark weight. MathTutorBench uses Overall. LongTutor combines Evidence with Teaching; Teaching is mapped from 1–5 to 0–100 using 25 × (T − 1). Scores are rounded only for display.</p></div></details>')
    output.append('<div class="data-links"><a href="data/omniedu-pro-scores.csv" download>Download scores · CSV ↓</a><a href="data/omniedu-pro-benchmarks.json" download>Raw benchmark metrics · JSON ↓</a><a href="https://huggingface.co/lhpku20010120/Omni-Edu-Pro-27B#evaluation">Pro model card ↗</a></div>')
    return '\n'.join(output)


def main():
    models = json.loads((ROOT / "data/omniedu-pro-benchmarks.json").read_text())["models"]
    for model in models:
        model["scores"] = scores(model["metrics"])
    page = ROOT / "index.html"
    content = page.read_text()
    start, end = '<!-- PRO_RESULTS_START -->', '<!-- PRO_RESULTS_END -->'
    if content.count(start) != 1 or content.count(end) != 1:
        raise ValueError("Expected exactly one Pro results block")
    before, remaining = content.split(start)
    _, after = remaining.split(end)
    page.write_text(before + start + '\n' + render(models) + '\n' + end + after)
    with (ROOT / "data/omniedu-pro-scores.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model"] + [title for _, title, _ in CATEGORIES])
        writer.writerows([m["name"]] + [fmt(s) for s in m["scores"]] for m in models)
    print(f"Updated {len(models)} models × 3 category scores.")


if __name__ == "__main__":
    main()
