# OmniEdu

## Open Foundation Models for Learning and Teaching

OmniEdu is an open family of education-oriented foundation models for **K–12 learning and teaching**. It is designed for the complete educational loop: understanding where a problem belongs in the curriculum, solving the problem, diagnosing a learner's difficulty, and selecting an effective instructional response.

🌐 [Project Page](https://haolpku.github.io/Omni-Edu/) · 🤗 [Dataset on Hugging Face](https://huggingface.co/datasets/lhpku20010120/Omni-Edu) · 📄 [Paper PDF](https://github.com/haolpku/Omni-Edu/blob/main/OmniEdu.pdf)

## Why OmniEdu?

Most general-purpose LLMs are optimized for answer generation. An educational model must also understand **what is being taught, what the learner already knows, and what to do next**. OmniEdu is trained around four complementary capabilities:

| Capability | What the model learns |
| --- | --- |
| Subject competence | Solve K–12 problems across mathematics, science, language, and other subjects |
| Curriculum grounding | Map questions, concepts, and solutions to fine-grained curriculum standards |
| Diagnostic reasoning | Identify misconceptions, missing prerequisites, and knowledge-state gaps |
| Pedagogical action | Explain, scaffold, give feedback, ask productive questions, and adapt instruction |

The released training corpus contains **69,999 instruction examples**, **15.96M supervised response tokens**, and **60,951 education-specific examples**. It combines curriculum resources, school-level problems, tutoring interactions, diagnostic tasks, and general instruction data in a capability-balanced mixture.

## Model family

| Model | Parameters | Best use |
| --- | ---: | --- |
| OmniEdu-4B | 4B | Low-cost local inference and classroom prototypes |
| OmniEdu-9B | 9B | Balanced deployment quality and serving cost |
| OmniEdu-27B | 27B | Highest-quality research and production evaluation |

All models use a 32K context window in training. The model checkpoints will be linked in this table as they are released.

## Results at a glance

OmniEdu improves over the corresponding untuned base models across curriculum grounding, K–12 problem solving, and pedagogical tutoring. OmniEdu-27B is among the strongest evaluated open-weight educational models and remains competitive with much larger proprietary systems.

| Benchmark | OmniEdu-27B | What it measures |
| --- | ---: | --- |
| K12-Bench | **63.12 EM / 76.69 F1** | Curriculum structure and concept grounding |
| MathFish | **85.89** | Alignment between problems and curriculum standards |
| EDUMATH | **86.95** | Curriculum-conditioned problem generation |
| GAOKAO-Bench | **94.87** | Authentic K–12 examination problem solving |
| MDK12-Bench | **57.76** | Multimodal and open-ended K–12 problem solving |
| MathTutorBench | **78.74 Scaffold WR** | Scaffolded mathematical tutoring |
| TutorBench | **59.42** | Adaptive explanation, feedback, and active learning |
| LongTutor | **78.20 Evidence / 3.02 Teaching** | Long-term, history-grounded tutoring |

The full model-by-model comparison and evaluation protocol are reported in the paper and supplementary tables.

## Quick start

### 1. Install inference dependencies

```bash
pip install -U torch transformers accelerate safetensors
```

### 2. Run with Transformers

Download a released checkpoint, or point `MODEL_ID` to a local checkpoint directory. The same script works for OmniEdu-4B, OmniEdu-9B, and OmniEdu-27B.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/path/to/OmniEdu-27B"  # replace with a released Hub ID when available

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {
        "role": "user",
        "content": "Explain why the seasons change, then give one misconception check for a Grade 7 student.",
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

answer = tokenizer.decode(
    outputs[0, inputs.shape[-1]:],
    skip_special_tokens=True,
)
print(answer)
```

### 3. Serve an OpenAI-compatible API with vLLM

```bash
pip install -U vllm

vllm serve /path/to/OmniEdu-27B \
  --served-model-name omniedu-27b \
  --dtype bfloat16 \
  --tensor-parallel-size 4 \
  --max-model-len 32768
```

For OmniEdu-4B or OmniEdu-9B, reduce `--tensor-parallel-size` to match the available GPUs. Then query the server with the standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="omniedu-27b",
    messages=[
        {"role": "user", "content": "Teach me how to solve x^2 - 5x + 6 = 0."}
    ],
    temperature=0.2,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## Dataset

The OmniEdu dataset is available on Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("lhpku20010120/Omni-Edu")
print(dataset)
```

The dataset is organized to support curriculum grounding, problem solving, diagnosis, tutoring, and general instruction. Please follow the licenses and usage terms of the component datasets and source materials.

## Intended use and limitations

OmniEdu is intended for research, educational prototyping, model evaluation, and teacher-assistance tools. It can produce incorrect or incomplete answers, and its outputs should be reviewed before use in high-stakes educational decisions. It should not be used as an autonomous authority for grading, placement, discipline, or student welfare decisions.

## Citation

```bibtex
@misc{omniedu2026,
  title  = {OmniEdu: Open Foundation Models for Learning and Teaching},
  author = {Liang, Hao and Lin, Qihan and Sun, Linzhuang and Zhang, Wentao},
  year   = {2026},
  url    = {https://github.com/haolpku/Omni-Edu}
}
```

## Acknowledgements

OmniEdu is built on open foundation models, open evaluation benchmarks, and community-maintained training tools. We thank the authors and maintainers of the underlying models, datasets, and LLaMA-Factory ecosystem.
