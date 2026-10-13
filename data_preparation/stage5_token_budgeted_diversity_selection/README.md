# Stage 5 — Token-budgeted diversity selection

K-center greedy selection inside each fine-grained task bucket (BGE-M3 embeddings), budgeted in supervised response tokens, over the pools filtered in stage 4. Produces the 60,951 education-specific examples. General-purpose data is not part of this stage.

```
stage5_token_budgeted_diversity_selection/
├── select_by_token_budget.py   # per-bucket k-center selection (budgets in TOKEN_BUDGETS)
├── validate_selection.py       # checks the selection against the paper's totals
└── results/bucketed_kcenter_final/   # selection output, input for stage 6
```

Candidates come from `stage4/results/quality_filtered_pools/`, and the bucket mapping is shared with stage 4 through `common/buckets.py`.

```bash
python data_preparation/stage5_token_budgeted_diversity_selection/select_by_token_budget.py \
  --embed-model models/bge-m3 --tokenizer models/Qwen3.5-9B-Base
python data_preparation/stage5_token_budgeted_diversity_selection/validate_selection.py
```
