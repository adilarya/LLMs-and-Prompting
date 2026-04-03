"""Task 3 – Dataset-Based Evaluation (Task 3C: Constrained Decoding and Structured Output).

Evaluates both LLMs on all 30 dataset examples using three prompting methods:
  zero_shot, few_shot (3-shot), and chain-of-thought (CoT).

Total outputs: 30 examples × 3 prompting methods × 2 models = 180 entries.

The results are saved in the assignment-required JSON schema to:
  results/csci5541-s26-hw5-adilarya-3c.json

Usage::

    python -m src.task3
    # or
    python src/task3.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_examples, get_few_shot_pool
from src.model_loader import MODELS, load_model, generate_chat, short_name
from utils.prompt_templates import (
    build_zero_shot_messages,
    build_few_shot_messages,
    build_cot_messages,
)
from utils.evaluation import (
    extract_answer,
    score_answer,
    check_format_valid,
    save_task3_json,
    compute_summary,
)


# Assignment-required output filename
OUTPUT_FILENAME = "csci5541-s26-hw5-adilarya-3c.json"

TASK_FAMILY = "3c"
DATASET_NAME = "custom_constrained_output_dataset"

# Generation config shared across all runs (greedy decoding for reproducibility)
_GEN_CONFIG_BASE = {
    "temperature": 0,
    "top_p": 1.0,
    "num_generations": 1,
    "seed": 42,
}

PROMPTING_METHODS = ["zero_shot", "few_shot", "cot"]

MAX_NEW_TOKENS = {
    "zero_shot": 60,
    "few_shot": 60,
    "cot": 150,
}


def _build_messages(question: str, method: str, demos: list) -> list:
    """Return chat messages for the given question and prompting method.

    Args:
        question: The question string.
        method: One of 'zero_shot', 'few_shot', 'cot'.
        demos: Few-shot demonstration examples (used only for 'few_shot').

    Returns:
        List of role/content message dicts.
    """
    if method == "zero_shot":
        return build_zero_shot_messages(question)
    if method == "few_shot":
        return build_few_shot_messages(question, demos)
    return build_cot_messages(question)


def run_task3(model_name: str, examples: list, demos: list) -> list:
    """Run all three prompting methods on 30 examples for a single model.

    Args:
        model_name: HuggingFace model identifier.
        examples: List of 30 example dicts from the dataset.
        demos: Few-shot demonstration pool (first 3 examples).

    Returns:
        List of assignment-schema result dicts (90 entries per model).
    """
    model, tokenizer = load_model(model_name)

    records = []
    for method in PROMPTING_METHODS:
        print(f"\n  [Model: {short_name(model_name)}  Method: {method}]")
        max_tokens = MAX_NEW_TOKENS[method]

        for example in examples:
            messages = _build_messages(example["question"], method, demos)
            raw_output = generate_chat(
                messages, model, tokenizer, max_new_tokens=max_tokens
            )
            predicted = extract_answer(raw_output, prompt_type=method)
            exact = score_answer(predicted, example["expected"])
            fmt_valid = check_format_valid(raw_output, prompt_type=method)

            print(
                f"    [{example['id']:02d}] expected={example['expected']!r:15s} "
                f"predicted={predicted!r:20s}  exact={exact}"
            )

            record = {
                "example_id": None,  # assigned after all records are collected
                "task_family": TASK_FAMILY,
                "dataset_name": DATASET_NAME,
                "dataset_item_id": f"ex_{example['id']:03d}",
                "model_name": model_name,
                "prompting_method": method,
                "messages": messages,
                "expected_output": example["expected"],
                "raw_model_output": raw_output,
                "scores": {
                    "exact_match": exact,
                    "format_valid": fmt_valid,
                },
                "annotation": {
                    "final_label": exact,
                    "notes": (
                        f"category={example['category']}; "
                        f"predicted={predicted!r}"
                    ),
                },
                "generation_config": {
                    **_GEN_CONFIG_BASE,
                    "max_new_tokens": max_tokens,
                },
            }
            records.append(record)

    return records


def main() -> None:
    """Entry point: run Task 3 for all models and save the assignment JSON."""
    print("=" * 70)
    print("Task 3 – Dataset-Based Evaluation (3C: Constrained Output)")
    print("=" * 70)
    print(f"Dataset : {DATASET_NAME}  ({len(load_examples())} examples)")
    print(f"Methods : {PROMPTING_METHODS}")
    print(f"Models  : {MODELS}")
    print(f"Expected outputs: {len(load_examples())} × {len(PROMPTING_METHODS)} × {len(MODELS)} = "
          f"{len(load_examples()) * len(PROMPTING_METHODS) * len(MODELS)}")

    examples = load_examples()
    demos = get_few_shot_pool(examples, n=3)

    all_records = []
    for model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}")
        print("=" * 70)
        records = run_task3(model_name, examples, demos)
        all_records.extend(records)

    # Assign sequential example_id values now that all records are collected
    for idx, record in enumerate(all_records, start=1):
        record["example_id"] = idx

    filepath = save_task3_json(all_records, OUTPUT_FILENAME)
    print(f"\n[task3] {len(all_records)} records saved to: {filepath}")

    # Print summary per model × method
    print("\n--- Accuracy Summary ---")
    print(f"{'Model':40s} {'Method':12s}  Accuracy")
    print("-" * 65)
    for model_name in MODELS:
        for method in PROMPTING_METHODS:
            subset = [
                r for r in all_records
                if r["model_name"] == model_name and r["prompting_method"] == method
            ]
            total = len(subset)
            correct = sum(r["scores"]["exact_match"] for r in subset)
            acc = round(correct / total, 3) if total else 0.0
            print(f"  {short_name(model_name):38s} {method:12s}  {correct}/{total}  ({acc:.1%})")


if __name__ == "__main__":
    main()
