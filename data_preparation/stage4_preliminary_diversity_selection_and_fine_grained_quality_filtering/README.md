# Stage 4 — Preliminary diversity selection and fine-grained quality filtering

Compresses sources whose candidate counts far exceed their budget with source-level k-center, then scores every candidate on the task-specific rubric (1-5) and filters, producing the candidate pools for stage 5.

```
stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering/
├── pools.py             # candidate assembly: stage 3 audited pools → pools/<capability>.jsonl
├── kcenter.py           # source-level k-center compression (BGE-M3, built-in targets)
├── rubrics.py           # rubric dimensions, critical dimensions and thresholds
├── score.py             # 1-5 scoring with the shared client (common/llm.py)
├── filter_by_scores.py  # rubric thresholds → results/quality_filtered_pools/
└── prompts/             # ten rubric prompts, each declaring its own output format
```

```bash
python data_preparation/stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering/pools.py
python data_preparation/stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering/kcenter.py --all
python data_preparation/stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering/score.py \
  --category <capability> --endpoints <scoring endpoints> --concurrency <value>
python data_preparation/stage4_preliminary_diversity_selection_and_fine_grained_quality_filtering/filter_by_scores.py
```

`score.py` takes the same connection options as stage 3 (`--endpoints`, `--model`, `--concurrency`, or `AUDIT_ENDPOINTS` / `AUDIT_MODEL`).

```
stage3/<capability>/<dataset>/audited/kept.jsonl           # input
stage4/pools/<capability>.jsonl                            # candidate pool (category, source, uid, kind)
stage4/preliminary/<source>.jsonl                          # source-level k-center result
stage4/results/scores.jsonl                                # per candidate: rubric id, scores, reason
stage4/results/quality_filtered_pools/<capability>.jsonl   # input for stage 5
```

## Decision rules

Rubric dimensions are scored 1-5; a candidate is kept only if every applicable dimension scores at least 3 and every critical dimension (correctness, validity, grounding) scores at least 4. Dimension lists and thresholds live in `rubrics.py`; RACE and C3 do not score `problem_validity`, and dimensions marked `null` by the prompt are treated as not applicable.
