# Stage 3 — LLM-assisted semantic auditing and rewriting

Audits every dataset with a served LLM (vLLM), deciding keep / rewrite / remove per example, then repairs the supervision of the examples routed to rewriting.

```
stage3_llm_assisted_semantic_auditing_and_rewriting/
├── sources.py    # dataset → prompt, declared checks and pool path
├── audit.py      # pool → per-example keep/rewrite/remove verdicts
├── rewrite.py    # rewrites the supervision of verdict=rewrite rows, then re-audits them
├── filter.py     # verdicts → audited pool (keep / remove / pending rewrite)
├── generate.py   # generation tasks (LongTutor synthetic, Eedi rationales)
└── prompts/      # all prompts; each one declares its own output format
```

The LLM client is shared with the other stages and lives in `common/llm.py`
(endpoint pool, concurrency, retries, resume).

## Connection parameters

Endpoints, model and concurrency depend on the deployment, so they are resolved as command line → environment variable → default; nothing is tied to one machine.

| Option | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--endpoints` | `AUDIT_ENDPOINTS` | `http://127.0.0.1:8400/v1` | vLLM endpoints, space- or comma-separated, round-robin over replicas |
| `--model` | `AUDIT_MODEL` | `Qwen3.5-122B` | served model name |
| `--concurrency` | — | endpoints × 32 | concurrent requests |
| `--max-tokens` / `--timeout` / `--attempts` | — | 8192 / 300 / 3 | generation cap, per-request timeout, retries |

## Usage

```bash
python data_preparation/stage3_llm_assisted_semantic_auditing_and_rewriting/audit.py \
  --dataset <dataset> --endpoints <vLLM endpoints> --concurrency <value>
python data_preparation/stage3_llm_assisted_semantic_auditing_and_rewriting/rewrite.py \
  --dataset <dataset> --endpoints <vLLM endpoints>
python data_preparation/stage3_llm_assisted_semantic_auditing_and_rewriting/filter.py --dataset <dataset>

python data_preparation/stage3_llm_assisted_semantic_auditing_and_rewriting/generate.py \
  --task LongTutor-Synthetic --input <pool> --field content
```

## Directory layout

```
<capability>/<dataset>/
├── cleaned/kept.jsonl          # stage 2 output, audit input
├── audit/verdicts.jsonl        # per example: verdict / usability_score / checks / reason
├── rewrite/rewritten.jsonl     # repaired supervision
├── rewrite/reaudit.jsonl       # verdicts of the repaired rows
└── audited/kept.jsonl          # audited pool, input for stage 4
```

## Decision rules

The model returns checklist judgements (yes/no per dimension, with `input_and_reference_valid` as the critical validity check) plus a 0-100 usability score; the action is derived by the pipeline:

- `input_and_reference_valid=no` or `usability_score < 50` → `remove`
- every checklist item yes and `usability_score >= 85` → `keep`
- otherwise (usable input, repairable supervision, or score 50-84) → `rewrite`

Each dataset's checklist lives in its own prompt file under `prompts/`; `sources.dimensions()` reads it from there and the code holds no second copy. Multimodal rows follow the same path: `image_ref` and `images` are rendered into the request.
