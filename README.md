# OmniEdu

## Open Foundation Models for Learning and Teaching

**Curriculum-aware, multimodal models for the full K–12 learning–teaching loop.** OmniEdu brings together subject knowledge, curriculum alignment, learner diagnosis, and instructional support in three model scales: **4B, 9B, and 27B**, each available in **BF16 and FP8**.

[📄 Paper (arXiv)](https://arxiv.org/abs/2609.23088) · [🌐 Project page](https://haolpku.github.io/Omni-Edu/) · [🤗 Models](#model-family) · [🤗 Dataset](https://huggingface.co/datasets/lhpku20010120/Omni-Edu) · [🛠️ Toolkit](#omniedu-toolkit)

Hao Liang · Qihan Lin · Meiyi Qiang · Linzhuang Sun · Hengyi Feng · Mingrui Chen · Sizhe Qiu · Wentao Zhang<br>
Peking University · University of the Chinese Academy of Sciences · Zhongguancun Academy

**October 2026:** FP8 checkpoints are available at all three scales, with **37%–44% smaller weight downloads** and per-benchmark BF16/FP8 comparisons. [Choose a checkpoint](#model-family) · [Compare FP8 results](#fp8-evaluation).

**September 2026:** Paper, model weights, and training data are available. Original BF16 results follow [arXiv v1, submitted September 19, 2026](https://arxiv.org/abs/2609.23088v1).

## Why OmniEdu?

An educational model needs to connect **what is being taught, what the learner understands, and what to do next**. OmniEdu organizes supervised fine-tuning around four complementary capabilities:

| Capability | What the model is trained to do |
| --- | --- |
| Subject competence | Solve K–12 problems across subjects and input formats |
| Curriculum grounding | Link questions, concepts, and solutions to curriculum standards and prerequisites |
| Diagnostic reasoning | Identify misconceptions, missing prerequisites, and gaps in learner understanding |
| Pedagogical action and scaffolding | Give targeted feedback, ask guiding questions, explain, and adapt instructional support |

The training mixture contains **69,999 instruction examples** and **15.96M supervised response tokens**, drawn from more than 100 sources: **60,951 education-specific examples** and **9,048 general-purpose examples**, with task-specific system instructions. The models use full-parameter supervised fine-tuning with a **32,768-token training sequence length**.

## Model family

| Model | BF16 weights | FP8 weights | Smaller download | Backbone reported in the paper |
| --- | --- | --- | ---: | --- |
| OmniEdu-4B | [BF16 · 8.46 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-4B) | [FP8 · 5.14 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-4B-FP8) | 39% | Qwen3.5-4B-Base |
| OmniEdu-9B | [BF16 · 17.53 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-9B) | [FP8 · 11.09 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-9B-FP8) | 37% | Qwen3.5-9B-Base |
| OmniEdu-27B | [BF16 · 50.96 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-27B) | [FP8 · 28.30 GiB](https://huggingface.co/lhpku20010120/Omni-Edu-27B-FP8) | 44% | Qwen3.8-27B |

Sizes are the summed `.safetensors` file sizes on Hugging Face (rounded, checked October 7, 2026), not total runtime GPU memory. Allow additional memory for the KV cache, activations, vision inputs, and serving overhead.

All six checkpoints use `Qwen3_5ForConditionalGeneration`. Choose BF16 for the original release or FP8 for a smaller checkpoint and quantized serving with a compatible backend. Training uses the `qwen3_5_nothink` template; keep `enable_thinking=False`.

### FP8 quantization

The FP8 releases use LLM Compressor and `compressed-tensors` with the `FP8_BLOCK` scheme: 128×128 block-wise FP8 weights and dynamic FP8 activations in groups of 128. The vision tower, linear-attention `in_proj_a/b` projections, and `lm_head` remain unquantized; other non-`Linear` modules retain their original dtype. The tokenizer, architecture, and chat template are unchanged. See each FP8 model card for its quantization recipe and configuration.

## Results at a glance

This section reports the original **BF16** checkpoints from the paper. Quantized-checkpoint results appear in [FP8 evaluation](#fp8-evaluation).

Across the paper's 16-model comparison, **OmniEdu-27B leads K12-Bench EM/F1, MathFish accuracy, and LongTutor's teaching score**. It reaches **86.95% EDUMATH MaC** and **78.74% MathTutorBench Scaffold win rate**, second among evaluated models on these two metrics. The comparisons include education-specific open-weight models and GPT-5.4, GPT-5.6-Sol, Claude-Opus-5, GLM-5.3, and Kimi-K3.

Each OmniEdu scale improves over its corresponding backbone on all summary metrics in the three education tables:

| Metric ↑ | OmniEdu-4B | OmniEdu-9B | OmniEdu-27B |
| --- | --- | --- | --- |
| K12-Bench EM | 54.25% | 55.46% | 63.12% |
| K12-Bench F1 | 71.75% | 73.68% | 76.69% |
| MathFish Acc. | 83.19% | 83.70% | 85.89% |
| EDUMATH MaC | 68.40% | 74.00% | 86.95% |
| GAOKAO-Bench Full | 90.46% | 93.66% | 94.87% |
| EXAMS-V Overall | 57.62% | 66.40% | 69.52% |
| MDK12-Bench Full | 46.36% | 50.80% | 57.76% |
| MathTutorBench Scaffold WR | 75.79% | 75.26% | 78.74% |
| MathTutorBench Scaffold-hard WR | 84.77% | 81.64% | 83.59% |
| TutorBench Overall | 46.67% | 48.16% | 59.42% |
| LongTutor Evidence | 65.88% | 66.63% | 78.20% |
| LongTutor Teaching | 2.29 | 2.66 | 3.02 |

### Full education comparisons

Scores reproduce arXiv v1 (Tables 1–3); higher is better. Values are percentages except LongTutor-T, which uses the original teaching-score scale. MathTutor-S / SH denote Scaffold / Scaffold-hard win rates; LongTutor-E / T denote Evidence / Teaching averages. The five education-specific baselines and GLM-5.3 receive text-only inputs on image-dependent examples, as described in the paper. Rankings apply to the evaluation protocol reported in the paper.

<details>
<summary>Curriculum grounding — all 16 evaluated models</summary>

| Model | Size | K12-Bench EM | K12-Bench F1 | MathFish Acc. | EDUMATH MaC |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B-Base | 4B | 42.72% | 69.20% | 80.33% | 47.60% |
| OmniEdu-4B (ours) | 4B | 54.25% | 71.75% | 83.19% | 68.40% |
| Qwen3.5-9B-Base | 9B | 48.52% | 71.99% | 79.66% | 58.60% |
| OmniEdu-9B (ours) | 9B | 55.46% | 73.68% | 83.70% | 74.00% |
| Qwen3.8-27B | 27B | 52.11% | 73.48% | 83.54% | 70.60% |
| OmniEdu-27B (ours) | 27B | 63.12% | 76.69% | 85.89% | 86.95% |
| Confucius3-Math | 14B | 5.23% | 19.23% | 0.00% | 3.00% |
| MuduoLLM | 14B | 48.24% | 70.46% | 82.85% | 57.20% |
| EduChat-SFT-Qwen2.5-7B | 7B | 45.20% | 67.58% | 78.23% | 28.00% |
| EduChat-R1-Qwen3-8B | 8B | 25.25% | 49.67% | 19.49% | 47.00% |
| EduChat-R1-Qwen3-32B | 32B | 49.42% | 71.21% | 83.71% | 61.20% |
| GPT-5.4 | – | 43.23% | 67.80% | 80.65% | 80.50% |
| GPT-5.6-Sol | – | 48.63% | 71.03% | 83.68% | 84.92% |
| Claude-Opus-5 | – | 48.48% | 71.34% | 83.52% | 75.00% |
| GLM-5.3 | – | 43.38% | 57.66% | 85.23% | 62.78% |
| Kimi-K3 | – | 55.03% | 74.71% | 85.40% | 90.00% |

[Source: Table 1](https://arxiv.org/html/2609.23088v1#S4.T1)

</details>

<details>
<summary>K–12 problem solving — all 16 evaluated models</summary>

| Model | Size | GAOKAO-Bench Full | EXAMS-V Overall | MDK12-Bench Full |
| --- | --- | --- | --- | --- |
| Qwen3.5-4B-Base | 4B | 88.98% | 44.69% | 35.43% |
| OmniEdu-4B (ours) | 4B | 90.46% | 57.62% | 46.36% |
| Qwen3.5-9B-Base | 9B | 92.94% | 63.00% | 44.50% |
| OmniEdu-9B (ours) | 9B | 93.66% | 66.40% | 50.80% |
| Qwen3.8-27B | 27B | 91.55% | 68.65% | 46.04% |
| OmniEdu-27B (ours) | 27B | 94.87% | 69.52% | 57.76% |
| Confucius3-Math | 14B | 84.51% | 21.95% | 39.13% |
| MuduoLLM | 14B | 89.66% | 21.95% | 45.44% |
| EduChat-SFT-Qwen2.5-7B | 7B | 70.75% | 21.95% | 32.50% |
| EduChat-R1-Qwen3-8B | 8B | 76.08% | 0.69% | 41.29% |
| EduChat-R1-Qwen3-32B | 32B | 83.86% | 21.95% | 41.84% |
| GPT-5.4 | – | 93.44% | 35.66% | 54.77% |
| GPT-5.6-Sol | – | 95.96% | 24.82% | 54.67% |
| Claude-Opus-5 | – | 97.22% | 64.34% | 57.46% |
| GLM-5.3 | – | 89.76% | 22.29% | 55.33% |
| Kimi-K3 | – | 94.86% | 87.29% | 63.60% |

[Source: Table 2](https://arxiv.org/html/2609.23088v1#S4.T2)

</details>

<details>
<summary>Pedagogical tutoring — all 16 evaluated models</summary>

| Model | Size | MathTutor-S | MathTutor-SH | TutorBench | LongTutor-E | LongTutor-T |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B-Base | 4B | 20.42% | 18.36% | 45.52% | 25.67% | 1.60 |
| OmniEdu-4B (ours) | 4B | 75.79% | 84.77% | 46.67% | 65.88% | 2.29 |
| Qwen3.5-9B-Base | 9B | 14.00% | 13.67% | 45.38% | 5.81% | 1.48 |
| OmniEdu-9B (ours) | 9B | 75.26% | 81.64% | 48.16% | 66.63% | 2.66 |
| Qwen3.8-27B | 27B | 57.16% | 55.86% | 58.58% | 36.80% | 2.74 |
| OmniEdu-27B (ours) | 27B | 78.74% | 83.59% | 59.42% | 78.20% | 3.02 |
| Confucius3-Math | 14B | 27.68% | 20.70% | 34.72% | 23.07% | 1.36 |
| MuduoLLM | 14B | 30.42% | 23.44% | 35.98% | 24.31% | 1.28 |
| EduChat-SFT-Qwen2.5-7B | 7B | 18.21% | 22.66% | 21.48% | 26.55% | 1.21 |
| EduChat-R1-Qwen3-8B | 8B | 11.58% | 7.42% | 23.71% | 33.75% | 1.14 |
| EduChat-R1-Qwen3-32B | 32B | 22.21% | 16.80% | 25.92% | 52.03% | 1.56 |
| GPT-5.4 | – | 6.32% | 1.96% | 46.34% | 75.50% | 1.62 |
| GPT-5.6-Sol | – | 10.53% | 17.65% | 50.97% | 77.50% | 1.89 |
| Claude-Opus-5 | – | 87.89% | 84.31% | 54.01% | 84.17% | 2.72 |
| GLM-5.3 | – | 77.89% | 76.92% | 38.62% | 77.42% | 2.66 |
| Kimi-K3 | – | 20.00% | 15.38% | 63.65% | 74.73% | 2.17 |

[Source: Table 3](https://arxiv.org/html/2609.23088v1#S4.T3)

</details>

### General capabilities after educational fine-tuning

The selected overall metrics below improve at all three scales. Individual submetrics can be flat or decline; see [Detailed results](https://arxiv.org/html/2609.23088v1#S6) for all IFEval, GPQA, and MMMU-Pro results.

| Model | IFEval Prompt Strict | GPQA Diamond | MMMU-Pro Overall |
| --- | --- | --- | --- |
| Qwen3.5-4B-Base | 64.88% | 58.08% | 50.46% |
| OmniEdu-4B (ours) | 65.06% | 60.61% | 52.60% |
| Qwen3.5-9B-Base | 69.69% | 62.12% | 58.38% |
| OmniEdu-9B (ours) | 73.01% | 63.64% | 60.75% |
| Qwen3.8-27B | 80.59% | 74.24% | 64.97% |
| OmniEdu-27B (ours) | 82.07% | 77.78% | 67.98% |

Sources: [IFEval, Table 13](https://arxiv.org/html/2609.23088v1#S6.T13), [GPQA, Table 14](https://arxiv.org/html/2609.23088v1#S6.T14), [MMMU-Pro, Table 15](https://arxiv.org/html/2609.23088v1#S6.T15). All values are percentages.

## FP8 evaluation

The following results are published in the FP8 model cards, alongside the original BF16 scores. They are separate from the paper's 16-model ranking. Quantization produces both increases and decreases across metrics; these are point estimates, not evidence of statistical equivalence.

### Curriculum grounding

| Model | K12-Bench EM | K12-Bench F1 | MathFish Acc. | EDUMATH MaC |
| --- | --- | --- | --- | --- |
| OmniEdu-4B BF16 | 54.25% | 71.75% | 83.19% | 68.40% |
| OmniEdu-4B FP8 | 53.69% | 70.94% | 82.26% | 67.64% |
| OmniEdu-9B BF16 | 55.46% | 73.68% | 83.70% | 74.00% |
| OmniEdu-9B FP8 | 55.62% | 73.45% | 81.77% | 73.77% |
| OmniEdu-27B BF16 | 63.12% | 76.69% | 85.89% | 86.95% |
| OmniEdu-27B FP8 | 63.16% | 76.59% | 85.13% | 86.84% |

### K–12 problem solving

| Model | GAOKAO-Bench Full | EXAMS-V Overall | MDK12-Bench Full |
| --- | --- | --- | --- |
| OmniEdu-4B BF16 | 90.46% | 57.62% | 46.36% |
| OmniEdu-4B FP8 | 89.47% | 57.14% | 47.30% |
| OmniEdu-9B BF16 | 93.66% | 66.40% | 50.80% |
| OmniEdu-9B FP8 | 91.09% | 63.83% | 49.20% |
| OmniEdu-27B BF16 | 94.87% | 69.52% | 57.76% |
| OmniEdu-27B FP8 | 95.72% | 69.15% | 58.81% |

### Pedagogical tutoring

| Model | MathTutor-S | MathTutor-SH | TutorBench | LongTutor-E | LongTutor-T |
| --- | --- | --- | --- | --- | --- |
| OmniEdu-4B BF16 | 75.79% | 84.77% | 46.67% | 65.88% | 2.29 |
| OmniEdu-4B FP8 | 76.63% | 83.20% | 46.15% | 66.11% | 2.30 |
| OmniEdu-9B BF16 | 75.26% | 81.64% | 48.16% | 66.63% | 2.66 |
| OmniEdu-9B FP8 | 78.11% | 82.42% | 48.01% | 69.04% | 2.76 |
| OmniEdu-27B BF16 | 78.74% | 83.59% | 59.42% | 78.20% | 3.02 |
| OmniEdu-27B FP8 | 76.11% | 80.08% | 59.34% | 78.95% | 3.05 |

Higher is better. Values are percentages except LongTutor-T (Teaching), which uses the original score scale. MathTutor-S / SH are Scaffold / Scaffold-hard win rates; LongTutor-E is Evidence.

Sources: [4B FP8 model card](https://huggingface.co/lhpku20010120/Omni-Edu-4B-FP8#evaluation) · [9B FP8 model card](https://huggingface.co/lhpku20010120/Omni-Edu-9B-FP8#evaluation) · [27B FP8 model card](https://huggingface.co/lhpku20010120/Omni-Edu-27B-FP8#evaluation).

## Quick start

### 1. Run with Transformers

For the original BF16 checkpoints, use a recent Transformers release with Qwen3.5 support and a GPU setup with sufficient memory for the chosen checkpoint. The example starts with 4B; change `MODEL_ID` for 9B or 27B.

```bash
pip install -U torch torchvision transformers accelerate safetensors pillow
```

```python
import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MODEL_ID = "lhpku20010120/Omni-Edu-4B"  # or Omni-Edu-9B / Omni-Edu-27B
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "A Grade 7 student says summer is warmer because Earth is closer to the Sun. Identify the misconception and give a guiding question before explaining.",
            }
        ],
    }
]
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

answer = processor.batch_decode(
    outputs[:, inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)[0]
print(answer)
```

For image-based questions, include an image item such as `{"type": "image", "image": "/path/to/question.png"}` alongside the text item in `content`. The processor prepares both modalities. See the [official Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) for supported input formats.

**Loading an FP8 checkpoint with Transformers:** install `compressed-tensors` in addition to the dependencies above and change `MODEL_ID` to an `-FP8` model ID. Transformers may decompress these weights during execution; a smaller download does not by itself guarantee lower inference memory. For serving this block-wise FP8 format, use the vLLM example below. See [Transformers loading modes](https://huggingface.co/docs/transformers/quantization/compressed_tensors#loading-modes).

### 2. Serve an OpenAI-compatible API with vLLM

Install a current vLLM release with Qwen3.5 support in a separate environment from the Transformers example, then launch a BF16 checkpoint:

```bash
pip install -U vllm

vllm serve lhpku20010120/Omni-Edu-4B \
  --served-model-name omniedu-4b \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Replace the model ID for 9B or 27B and set `--tensor-parallel-size` to the number of GPUs used for the model. If you change the model size, also change `--served-model-name` to `omniedu-9b` or `omniedu-27b` so it matches the corresponding SDK alias. Required GPU memory also depends on context length, concurrency, and vision inputs; reduce `--max-model-len` if needed. The serving configuration follows the [official vLLM Qwen3.5 guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

#### Serve the pre-quantized FP8 weights

On a GPU/backend that supports block-wise FP8, launch the published FP8 checkpoint directly:

```bash
vllm serve lhpku20010120/Omni-Edu-4B-FP8 \
  --served-model-name omniedu-4b \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Replace the ID with `lhpku20010120/Omni-Edu-9B-FP8` or `lhpku20010120/Omni-Edu-27B-FP8` as needed, and update the served model name accordingly. vLLM reads the checkpoint's `compressed-tensors` quantization configuration automatically; `--dtype bfloat16` sets the compute dtype for unquantized components and does not convert the stored FP8 weights to BF16. Set GPU parallelism and context length to fit your hardware. Both BF16 and FP8 servers use the same API request below.

Query the running server from another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "omniedu-4b",
    "messages": [{"role": "user", "content": "Help a student solve x^2 - 5x + 6 = 0. Start with a hint."}],
    "temperature": 0.2,
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

## Dataset

Load the released instruction mixture from [Hugging Face](https://huggingface.co/datasets/lhpku20010120/Omni-Edu):

```bash
pip install -U datasets
```

```python
from datasets import load_dataset

train = load_dataset(
    "lhpku20010120/Omni-Edu",
    "core_v6_full_system_prompted",
    split="train",
)
print(train)
```

The dataset covers curriculum grounding, problem solving, diagnosis, tutoring, and general instruction. Please follow the licenses and usage terms of the component datasets and source materials.

## Code

This repository includes the data preparation and training code, along with a Python toolkit for using and evaluating OmniEdu models.

| Directory | Description |
| --- | --- |
| [`data_preparation/`](data_preparation/) | Six-stage data preparation pipeline, from source collection to the final instruction dataset |
| [`training/`](training/) | Full-parameter supervised fine-tuning of the 4B, 9B, and 27B models using LLaMA-Factory |
| [`sdk/`](sdk/) | The `omniedu` toolkit for model inference, multimodal chat, teaching instructions, batch processing, and benchmark evaluation |

### OmniEdu toolkit

The [`omniedu` Python package](https://pypi.org/project/omniedu/) provides a unified interface for OmniEdu models and other models served through OpenAI-compatible APIs. Its evaluation suite supports **12 educational and general-purpose benchmarks**, with tools for scoring open-ended responses and generating Markdown comparison tables.

Install from PyPI:

```bash
pip install omniedu
```

With the 4B vLLM service launched above, you can make a request through the SDK:

```python
from omniedu import OmniEdu

model = OmniEdu.from_pretrained("4B", base_url="http://127.0.0.1:8000/v1")
response = model.chat(text="Help a student solve x^2 - 5x + 6 = 0. Start with a hint.")
print(response.text)
```

To explore the evaluation suite:

```bash
pip install "omniedu[eval]"
omniedu eval --list
```

See the **[toolkit README](sdk/README.md)** for image inputs, task-specific instructions, JSONL batch inference, evaluation of your own models, and report generation. Benchmark datasets and upstream code must be obtained separately where required.

For training, see the **[training README](training/README.md)**. To inspect the configuration without launching a run:

```bash
bash training/run_training.sh 9b --dry-run
```

The code is licensed under Apache-2.0. Model weights and third-party datasets and benchmarks retain their respective licenses.

## Intended use and limitations

OmniEdu supports research, educational prototypes, and teacher-assistance tools. Benchmark performance does not establish classroom learning gains. Models can give incorrect answers or unsuitable guidance; educators should review outputs before consequential use. The paper discusses evaluation coverage, multimodal limitations, and deployment considerations in more detail.

## Citation

```bibtex
@misc{liang2026omniedu,
  title         = {OmniEdu: Open Foundation Models for Learning and Teaching},
  author        = {Hao Liang and Qihan Lin and Meiyi Qiang and Linzhuang Sun and Hengyi Feng and Mingrui Chen and Sizhe Qiu and Wentao Zhang},
  year          = {2026},
  eprint        = {2609.23088},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.23088}
}
```

## Acknowledgements

We thank the authors and maintainers of the underlying models, datasets, benchmarks, and the LLaMA-Factory ecosystem.
