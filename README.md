# LLMs and Prompting

A Python project that evaluates three prompting techniques — **zero-shot**, **few-shot**, and **chain-of-thought (CoT)** — on two open-weight HuggingFace language models in the 1 B – 3 B parameter range.

| Model | Parameters |
|---|---|
| `EleutherAI/gpt-neo-1.3B` | ~1.3 B |
| `EleutherAI/gpt-neo-2.7B` | ~2.7 B |

Results are scored with substring-based exact match and saved as structured JSON files.

---

## Project structure

```
LLMs-and-Prompting/
├── data/
│   ├── dataset_loader.py   # Load examples; split few-shot pool / eval set
│   └── examples.json       # 30 QA examples (factual, math, reasoning)
├── src/
│   ├── model_loader.py     # Load models & generate text (shared utilities)
│   ├── task1.py            # Zero-shot experiments
│   ├── task2.py            # Few-shot experiments (3-shot)
│   └── task3.py            # Chain-of-thought experiments
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
> `float16` and run on GPU automatically.  On CPU-only machines inference is
> slower but fully functional.

---

## Running the experiments

Each task script can be run from the **project root**:

```bash
# Task 1 – Zero-shot prompting
python src/task1.py

# Task 2 – Few-shot prompting (3-shot)
python src/task2.py

# Task 3 – Chain-of-thought prompting
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

Every run produces a JSON file at `results/<experiment>_<model_slug>.json`
with the following schema:

```json
{
  "experiment": "task1_zero_shot",
  "model": "EleutherAI/gpt-neo-1.3B",
  "timestamp": "2026-04-02T20:21:34",
  "results": [
    {
      "id": 0,
      "question": "What is the capital of France?",
      "expected": "Paris",
      "category": "factual",
      "prompt": "Answer the following question concisely.\n\nQuestion: ...\nAnswer:",
      "output": "Paris",
      "predicted": "Paris",
      "score": 1
    }
  ],
  "summary": {
    "total": 30,
    "correct": 18,
    "accuracy": 0.6
  }
}
```

`score` is `1` if the expected answer appears (case-insensitively) anywhere in
the predicted answer, otherwise `0`.

---

## Dataset

`data/examples.json` contains 30 examples spanning three categories:

| Category | Count | Description |
|---|---|---|
| `factual` | 13 | Geography, science, history |
| `math` | 13 | Arithmetic and word problems |
| `reasoning` | 4 | Logic and probability |

For **few-shot** experiments (Task 2) the first 3 examples are used as
in-context demonstrations and the remaining 27 are evaluated.

---

## Reproducibility

* Greedy decoding (`temperature=0`) is used for all experiments.
* The few-shot demonstration pool is always the first 3 examples of
  `examples.json`.
* Model weights are downloaded from the HuggingFace Hub on first run and
  cached locally (usually `~/.cache/huggingface/`).
