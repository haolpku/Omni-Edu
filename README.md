# OmniEdu

## Open Foundation Models for Learning and Teaching

**Curriculum-aware, multimodal models for the full K–12 learning–teaching loop.** OmniEdu brings together subject knowledge, curriculum alignment, learner diagnosis, and instructional support in three released checkpoints: **4B, 9B, and 27B**.

[📄 Paper (arXiv)](https://arxiv.org/abs/2609.23088) · [🌐 Project page](https://haolpku.github.io/Omni-Edu/) · [🤗 Models](#model-family) · [🤗 Dataset](https://huggingface.co/datasets/lhpku20010120/Omni-Edu)

Hao Liang · Qihan Lin · Meiyi Qiang · Linzhuang Sun · Hengyi Feng · Mingrui Chen · Sizhe Qiu · Wentao Zhang<br>
Peking University · University of the Chinese Academy of Sciences · Zhongguancun Academy

**September 2026:** Paper, model weights, and training data are available. Results below follow [arXiv v1, submitted September 19, 2026](https://arxiv.org/abs/2609.23088v1).

## Why OmniEdu?

An educational model needs to connect **what is being taught, what the learner understands, and what to do next**. OmniEdu organizes supervised fine-tuning around four complementary capabilities:

| Capability | What the model is trained to do |
| --- | --- |
| Subject competence | Solve K–12 problems across subjects and input formats |
| Curriculum grounding | Link questions, concepts, and solutions to curriculum standards and prerequisites |
| Diagnostic reasoning | Identify misconceptions, missing prerequisites, and gaps in learner understanding |
| Pedagogical action and scaffolding | Give targeted feedback, ask guiding questions, explain, and adapt instructional support |

The training mixture contains **69,999 instruction examples** and **15.96M supervised response tokens**, drawn from more than 100 sources: **60,951 education-specific examples** and **9,048 general-purpose examples**, with **20 task-specific system instructions**. The models use full-parameter supervised fine-tuning with a **32,768-token training sequence length**.

## Model family

| Checkpoint | Scale | Backbone reported in the paper |
| --- | ---: | --- |
| [OmniEdu-4B](https://huggingface.co/lhpku20010120/Omni-Edu-4B) | 4B | Qwen3.5-4B-Base |
| [OmniEdu-9B](https://huggingface.co/lhpku20010120/Omni-Edu-9B) | 9B | Qwen3.5-9B-Base |
| [OmniEdu-27B](https://huggingface.co/lhpku20010120/Omni-Edu-27B) | 27B | Qwen3.8-27B |

All three checkpoints are downloadable on Hugging Face. Their published configurations use `Qwen3_5ForConditionalGeneration`; use the multimodal processor and model class in the example below. Training uses the `qwen3_5_nothink` template, so the examples explicitly disable thinking.

## Results at a glance

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

## Quick start

### 1. Run with Transformers

Use a recent Transformers release with Qwen3.5 support and a GPU setup with sufficient memory for the chosen checkpoint. The example starts with 4B; change `MODEL_ID` for 9B or 27B.

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

### 2. Serve an OpenAI-compatible API with vLLM

Install a current vLLM release with Qwen3.5 support in a separate environment from the Transformers example, then launch a checkpoint:

```bash
pip install -U vllm

vllm serve lhpku20010120/Omni-Edu-4B \
  --served-model-name omniedu \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

Replace the model ID for 9B or 27B and set `--tensor-parallel-size` to the number of GPUs used for the model. Required GPU memory also depends on context length, concurrency, and vision inputs; reduce `--max-model-len` if needed. The serving configuration follows the [official vLLM Qwen3.5 guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

Query the running server from another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "omniedu",
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
