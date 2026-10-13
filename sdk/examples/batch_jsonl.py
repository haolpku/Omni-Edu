"""JSONL in, JSONL out — the reproducibility path."""

from omniedu import OmniEdu

model = OmniEdu.from_pretrained(
    "4B",
    base_url="http://127.0.0.1:8000/v1",
    concurrency=32,
)

summary = model.batch(
    "problems.jsonl",
    out="predictions.jsonl",
    task="solve_reasoned",
)
print(summary)

model.close()

