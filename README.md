# LLMs and Prompting

A Python project that evaluates three prompting techniques — **zero-shot**, **few-shot (3-shot)**, and **chain-of-thought (CoT)** — on two open-weight HuggingFace instruction-tuned language models in the 1 B – 3 B parameter range.

| Model | Family | Parameters |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | Meta LLaMA | ~3.21 B |
| `Qwen/Qwen2.5-3B-Instruct` | Alibaba Qwen | ~3.09 B |

The selected models are similar in size (~3 B parameters) but come from **different model families**, enabling a fair comparison of architecture, tokenizer, and instruction-following behavior.

This project focuses on **Task 3C: Constrained Decoding and Structured Output**. Results are evaluated using exact-match accuracy and format validity, and saved as structured JSON files following the assignment schema.

---

## Project structure

```
LLMs-and-Prompting/
├── data/
│   ├── dataset_loader.py   # Load examples; split few-shot pool / eval set
│   └── examples.json       # 30 QA examples (factual, math, reasoning)
├── src/
│   ├── model_loader.py     # Load instruction-tuned models & generate text
│   ├── task1.py            # Task 1 – Model Selection + 5 Original Examples
│   ├── task2.py            # Task 2 – Prompting Techniques (3 methods × 2 models)
│   └── task3.py            # Task 3 – Dataset Evaluation (180 outputs)
├── utils/
│   ├── prompt_templates.py # Build zero-shot / few-shot / CoT chat messages
│   └── evaluation.py       # Answer extraction, scoring, JSON writers
├── results/                # Output JSON files (created at runtime)
│   └── csci5541-s26-hw5-adilarya-3c.json  # Task 3 assignment submission file
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/adilarya/LLMs-and-Prompting.git
cd LLMs-and-Prompting
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** If a CUDA-capable GPU is available, models are loaded in
> `float16` and run on GPU automatically. On CPU-only machines inference is
> slower but fully functional.

### 4. HuggingFace access

`meta-llama/Llama-3.2-3B-Instruct` requires accepting Meta's license on
HuggingFace and setting your access token:

```bash
huggingface-cli login
```

Or set the environment variable:

```bash
export HF_TOKEN=<your_token>
```

---

## Running the experiments

Each task script can be run from the **project root**:

```bash
# Task 1 – Model Selection + 5 Original Examples
python src/task1.py

# Task 2 – Prompting Techniques (zero-shot, few-shot, CoT on 5 examples)
python src/task2.py

# Task 3 – Dataset Evaluation (30 examples × 3 methods × 2 models = 180 outputs)
python src/task3.py
```

You can also invoke them as modules:

```bash
python -m src.task1
python -m src.task2
python -m src.task3
```

Each script loops over both models, prints per-example predictions to stdout,
and writes results to the `results/` directory.

---

## Output format

### Internal results (Task 1 & 2)

Tasks 1 and 2 write a JSON file with an internal schema containing the
experiment name, model, timestamp, per-example results, and aggregate summary.

### Assignment submission (Task 3)

Task 3 writes `results/csci5541-s26-hw5-adilarya-3c.json` following the
assignment-required schema. Each entry corresponds to a single model run on a
single example under a single prompting method:

```json
[
  {
    "example_id": 1,
    "task_family": "3c",
    "dataset_name": "custom_constrained_output_dataset",
    "dataset_item_id": "ex_000",

    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "prompting_method": "zero_shot",

    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Answer questions concisely."},
      {"role": "user", "content": "Answer the following question concisely.\n\nQuestion: What is the capital of France?"}
    ],

    "expected_output": "Paris",
    "raw_model_output": "Paris",

    "scores": {
      "exact_match": 1,
      "format_valid": 1
    },

    "annotation": {
      "final_label": 1,
      "notes": "category=factual; predicted='Paris'"
    },

    "generation_config": {
      "temperature": 0,
      "top_p": 1.0,
      "max_new_tokens": 60,
      "num_generations": 1,
      "seed": 42
    }
  }
]
```

The full evaluation consists of **30 examples × 3 prompting methods × 2 models = 180 total outputs**.

---

## Dataset

The dataset (`data/examples.json`) consists of 30 curated examples designed to
evaluate structured output and reasoning ability. The examples span:

- Factual questions (e.g., geography, science)
- Mathematical reasoning problems
- Logical reasoning tasks

The same 30 examples are used across all models and prompting methods to
ensure a fair comparison.

---

## Prompting Techniques

Three prompting strategies are evaluated:

| Method | Description | Suitable for | Limitations |
|---|---|---|---|
| **Zero-shot** | Model receives only the task instruction, no examples | Well-known tasks with simple outputs | May fail on novel formats or nuanced constraints |
| **Few-shot (3-shot)** | 3 in-context demonstrations guide response style and format | Structured / constrained output tasks | Can overfit to demo style; doesn't generalize well out-of-distribution |
| **Chain-of-thought (CoT)** | Model reasons step-by-step before stating the final answer | Multi-step reasoning and math | Higher latency; may over-generate on simple questions |

---

## Reproducibility

* Greedy decoding (`temperature=0`) is used for all experiments.
* Models are loaded with `torch_dtype=float16` on GPU and `float32` on CPU.
* The few-shot demonstration pool always uses the first 3 examples of `examples.json`.
* Model weights are downloaded from the HuggingFace Hub on first run and cached locally (usually `~/.cache/huggingface/`).
* All prompts use the model's `apply_chat_template` for correct special-token formatting.
