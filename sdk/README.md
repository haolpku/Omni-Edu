# OmniEdu

**A unified inference and evaluation toolkit for the Omni-Edu model family.**

[Omni-Edu](https://huggingface.co/OpenDCAI) is an open-source model family for K–12 learning and teaching, available in 4B, 9B, and 27B sizes. The models are fine-tuned from the Qwen3.5/3.8 series for educational tasks.

`omniedu` provides a Python API and command-line tools for text and image inputs, task-specific teaching instructions, and batch inference. It also offers a unified interface for running benchmarks on both Omni-Edu and other models served through OpenAI-compatible APIs.

| Feature | Interface |
| --- | --- |
| Model inference and multimodal chat | `OmniEdu.from_pretrained()`, `model.chat()` |
| Teaching task instructions | `model.chat(..., task=...)` |
| JSONL batch inference | `omniedu batch` |
| Benchmark evaluation | `omniedu eval` |
| Evaluation report generation | `omniedu report` |
| Open-ended response scoring | `omniedu judge` |

## Installation

Choose the installation option that fits your use case:

```bash
pip install omniedu                    # Connect to OpenAI-compatible inference services
pip install "omniedu[transformers]"    # Load models locally
pip install "omniedu[eval]"            # Run benchmark evaluations
pip install "omniedu[eval,eval-local]" # Add dependencies for local scoring
```

Model weights are hosted on Hugging Face and are not included in the PyPI package. Third-party benchmark code and datasets must be obtained separately under their respective terms.

## Quick Start

### Model Inference

If you already have an OpenAI-compatible inference service running, connect to it directly:

```python
from omniedu import OmniEdu

model = OmniEdu.from_pretrained("27B", base_url="http://127.0.0.1:8000/v1")

response = model.chat(
    text="A student expanded 2(x + 3) as 2x + 3. Identify the mistake and give a hint first.",
    task="diagnose_and_correct",
)
print(response.text)
```

Replace `"27B"` with `"4B"` or `"9B"` to use a different Omni-Edu model size. You can also provide a full Hugging Face model ID or a local model path.

To include an image, pass it using the `image` argument:

```python
response = model.chat(
    text="Review this student's work. Identify any errors and provide step-by-step hints.",
    image="student_work.png",
    task="diagnose_and_correct",
)
print(response.text)
```

You can also pass multi-turn conversations through `messages` or use `await model.achat(...)` for asynchronous inference.

### Serving a Model

You can serve Omni-Edu with [vLLM](https://github.com/vllm-project/vllm):

```bash
vllm serve OpenDCAI/Omni-Edu-27B \
  --served-model-name omniedu-27b \
  --max-model-len 32768 \
  --tensor-parallel-size 4 \
  --limit-mm-per-prompt '{"image": 12}' \
  --trust-remote-code
```

Adjust the deployment settings to match your hardware. The toolkit can also generate the corresponding launch command:

```bash
omniedu serve --model 27B --tensor-parallel-size 4
```

**Note:** `omniedu serve` prints the command; it does not start the server.

When connecting to a remote service using the short aliases `"4B"`, `"9B"`, or `"27B"`, the SDK automatically uses the corresponding served model names (`omniedu-4b`, `omniedu-9b`, or `omniedu-27b`). If your service uses a different name, specify it with `served_model=`. Full model IDs and local paths are not rewritten.

Alternatively, install `omniedu[transformers]` and use `OmniEdu.from_pretrained("4B")` to load a model locally without a separate inference server.

## Teaching Task Instructions

OmniEdu includes 19 teaching task instructions covering problem solving, error diagnosis, knowledge-point analysis, and guided tutoring. Select an instruction with the `task` argument:

```python
response = model.chat(
    text="Don't give the answer directly. Guide the student with questions instead.",
    task="multiturn_socratic",
)
```

| Use case | Task ID |
| --- | --- |
| Step-by-step solutions | `solve_reasoned` |
| Error diagnosis and feedback | `diagnose_and_correct` |
| Knowledge-point analysis | `knowledge_point` |
| Socratic tutoring | `multiturn_socratic` |

Use `list_tasks()` to explore all 19 built-in tasks. You can register custom instructions with `register_task(id, instruction)` or provide a system instruction directly using `system=`.

**Task instructions are optional.** If neither `task` nor `system` is provided, no additional system instruction is added. **You do not need to specify `task` when running benchmarks.**

## Batch Inference

Use `omniedu batch` to process JSONL files with concurrent requests and resume support:

```bash
omniedu batch \
  --input problems.jsonl \
  --output predictions.jsonl \
  --model 27B \
  --base-url http://127.0.0.1:8000/v1 \
  --task solve_reasoned \
  --concurrency 32
```

Input records can use the `messages` conversation format or simpler fields such as `text`, `prompt`, or `question`, with optional `image` or `images` fields. When restarted, the command skips completed records and retries unfinished or failed ones.

## Model Evaluation

`omniedu eval` provides a unified interface for 12 educational and general-purpose benchmarks. You can evaluate Omni-Edu models or other models deployed through an OpenAI-compatible API.

Supported benchmarks: **K12-Bench, MathFish, EDUMATH, GAOKAO-Bench, EXAMS-V, MDK12-Bench, MathTutorBench, TutorBench, LongTutor, IFEval, GPQA, and MMMU-Pro**.

### Running Evaluations

```bash
# List supported benchmarks
omniedu eval --list

# Evaluate Omni-Edu-4B
omniedu eval \
  --model 4B \
  --base-url http://127.0.0.1:8000/v1 \
  --benchmarks gpqa,ifeval,MMMU-Pro

# Evaluate your own model (use its served model name)
omniedu eval \
  --model my-model \
  --base-url http://127.0.0.1:8000/v1 \
  --benchmarks gpqa,ifeval,MMMU-Pro
```

Use `--benchmarks` to select one or more benchmarks, or `all` to run the full suite.

Some benchmarks require upstream datasets or code to be prepared in advance. Use `--official-root` to specify the corresponding repository path when needed. For example:

```bash
omniedu eval \
  --model 27B \
  --base-url http://127.0.0.1:8000/v1 \
  --benchmarks K12-Bench \
  --official-root K12-Bench=/data/K12-KGraph \
  --data-root evaluation_data
```

### Generating Reports

Each benchmark produces a `summary.json` file. Use `omniedu report` to combine results from multiple models into Markdown tables following the format of the three main results tables in the [Omni-Edu paper](https://arxiv.org/abs/2609.23088). You can also export the aggregated metrics as JSON:

```bash
omniedu report \
  --results-root results/base/outputs \
  --results-root results/my_sft/outputs \
  --out tables.md \
  --json metrics.json
```

If scoring outputs are stored separately, use `--judgments-root` to specify their location.

### Scoring Open-Ended Responses

Some benchmarks require an additional scoring step for open-ended responses. Use `omniedu judge` with a separate judge model:

```bash
omniedu judge \
  --task tutorbench \
  --input predictions.jsonl \
  --output judged.jsonl \
  --base-url https://api.example.com/v1 \
  --judge-model gpt-4.1-mini
```

Here, `--base-url` and `--judge-model` refer to the scoring service, which may differ from the model being evaluated.

## License

The toolkit is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Model weights, third-party benchmark code, and datasets remain subject to their respective licenses and terms of use.
