# LLMs and Prompting

A Python project that evaluates three prompting techniques — **zero-shot**, **few-shot**, and **chain-of-thought (CoT)** — on two open-weight HuggingFace language models in the 1 B – 3 B parameter range.

| Model | Parameters |
|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | ~3.21 B |
| `Qwen/Qwen2.5-3B-Instruct` | ~3.09 B |

Results are evaluated using exact match accuracy and format validity, and saved as structured JSON files following the assignment schema. The selected models are similar in size (~3B parameters) but come from different model families, enabling a fair comparison of architecture and prompting behavior. This project focuses on Task 3C: Constrained Decoding and Structured Output.

---

## Project structure

```
LLMs-and-Prompting/
├── data/
│   ├── dataset_loader.py   # Load examples; split few-shot pool / eval set
│   └── examples.json       # 30 QA examples (factual, math, reasoning)
├── src/
│   ├── model_loader.py     # Load models & generate text (shared utilities)
│   ├── task1.py            # Model Selection + 5 Examples
│   ├── task2.py            # Prompting Techniques
│   └── task3.py            # Dataset Evaluation
├── utils/
│   ├── prompt_templates.py # Build zero-shot / few-shot / CoT prompts
│   └── evaluation.py       # Answer extraction, scoring, JSON writer
├── results/                # Output JSON files (created at runtime)
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

> **GPU note:** If a CUDA-capable GPU is available the models will be loaded in
> `float16` and run on GPU automatically. On CPU-only machines inference is
> slower but fully functional.

---

## Running the experiments

Each task script can be run from the **project root**:

```bash
# Task 1 – Model Selection + 5 Examples
python src/task1.py

# Task 2 – Prompting Techniques
python src/task2.py

# Task 3 – Dataset Evaluation
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

Results are saved in a JSON file following the assignment-required schema. Each entry corresponds to a single model run on a single example under a single prompting method.

Example format:

```json
[
  {
    "example_id": 1,
    "task_family": "3c",
    "dataset_name": "custom_dataset",
    "dataset_item_id": "ex_001",

    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    "prompting_method": "few_shot",

    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Prompt text here"}
    ],

    "expected_output": "Paris",
    "raw_model_output": "Paris",

    "scores": {
      "exact_match": 1,
      "format_valid": 1
    },

    "annotation": {
      "final_label": 1,
      "notes": "Correct answer"
    },

    "generation_config": {
      "temperature": 0,
      "top_p": 1.0,
      "max_new_tokens": 50,
      "num_generations": 1,
      "seed": 42
    }
  }
]
```

Scores are computed using exact match and format validity. Exact match checks whether the predicted output matches the expected answer, while format validity ensures the output follows the required structure. The full evaluation consists of 30 examples × 3 prompting methods × 2 models = 180 total outputs.

---

## Dataset

The dataset consists of 30 curated examples designed to evaluate structured output and reasoning ability. The examples span:

- factual questions (e.g., geography, science)
- mathematical reasoning problems
- logical reasoning tasks

The dataset is constructed to provide clear expected answers, enabling exact-match evaluation. The same 30 examples are used across all models and prompting methods to ensure a fair comparison.

--- 

## Prompting Techniques

We evaluate three prompting strategies:

- **Zero-shot**: The model is given only the task prompt without examples. Serves as a baseline.
- **Few-shot**: The model is provided with a small number of examples (3-shot) to guide its response.
- **Chain-of-thought (CoT)**: The model is prompted to reason step-by-step before producing a final answer.

These methods are chosen to compare baseline performance, in-context learning, and structured reasoning behavior.

---

## Reproducibility

* Greedy decoding (`temperature=0`) is used for all experiments.
* The few-shot demonstration pool is always the first 3 examples of
  `examples.json`.
* Model weights are downloaded from the HuggingFace Hub on first run and
  cached locally (usually `~/.cache/huggingface/`).
