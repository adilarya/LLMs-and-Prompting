"""Task 2 – Prompting Techniques Comparison.

Applies three prompting methods — zero-shot, few-shot (3-shot), and
chain-of-thought (CoT) — to the five original examples from Task 1.
Both models are evaluated for each method, and the results are compared.

Prompting Methods:
  1. Zero-shot: The model receives only the task instruction with no examples.
     Suitable for well-known tasks; can fail on novel or nuanced problems.
  2. Few-shot (3-shot): Three in-context examples guide the model's response
     style and format. Improves structured output but may overfit to demo style.
  3. Chain-of-thought (CoT): The model is asked to reason step-by-step before
     answering. Helps with multi-step reasoning; adds latency and may
     over-generate on simple questions.

Usage::

    python -m src.task2
    # or
    python src/task2.py

Results are written to results/task2_prompting_techniques.json.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import MODELS, load_model, generate_chat
from src.task1 import TASK1_EXAMPLES, evaluate_example
from utils.prompt_templates import (
    build_zero_shot_messages,
    build_few_shot_messages,
    build_cot_messages,
)
from utils.evaluation import save_results


# Fixed 3-shot demonstrations
_DEMO_POOL = [
    {
        "question": (
            "Classify the sentiment of the following review as exactly one of: "
            "Positive, Negative, or Neutral.\n\n"
            "Review: \"Absolutely loved the product, fast shipping!\""
        ),
        "expected": "Positive",
    },
    {
        "question": (
            "Answer with only the letter of the correct option.\n\n"
            "Which element has the chemical symbol 'O'?\n"
            "A) Gold\nB) Oxygen\nC) Osmium\nD) Oganesson"
        ),
        "expected": "B",
    },
    {
        "question": (
            "Name the country described by the clue below. "
            "Give only the country name.\n\n"
            "Clue: \"This country is shaped like a boot and is home to the Colosseum.\""
        ),
        "expected": "Italy",
    },
]

PROMPTING_METHODS = {
    "zero_shot": build_zero_shot_messages,
    "few_shot": lambda q: build_few_shot_messages(q, _DEMO_POOL),
    "cot": build_cot_messages,
}

MAX_NEW_TOKENS = {
    "zero_shot": 80,
    "few_shot": 80,
    "cot": 200,
}


def run_task2(model_name: str) -> list:
    """Run all three prompting methods on the 5 Task 1 examples for one model."""
    model, tokenizer = load_model(model_name)

    results = []
    for method_name, builder in PROMPTING_METHODS.items():
        print(f"\n  [Method: {method_name}]")
        for ex in TASK1_EXAMPLES:
            messages = builder(ex["question"])
            raw_output = generate_chat(
                messages,
                model,
                tokenizer,
                max_new_tokens=MAX_NEW_TOKENS[method_name],
            )

            predicted, correct, note = evaluate_example(ex, raw_output)

            print(
                f"    Ex {ex['id']:02d}: expected={ex['expected']!r:15s} "
                f"predicted={predicted[:40]!r}  "
                f"[{'OK' if correct else 'WRONG'}]  {note}"
            )

            results.append(
                {
                    "id": ex["id"],
                    "task": ex["task"],
                    "question": ex["question"],
                    "expected": ex["expected"],
                    "category": ex["category"],
                    "model": model_name,
                    "prompting_method": method_name,
                    "messages": messages,
                    "raw_output": raw_output,
                    "predicted": predicted,
                    "correct": correct,
                    "note": note,
                }
            )
    return results


def main() -> None:
    """Entry point: run Task 2 for all models and prompting methods."""
    print("=" * 70)
    print("Task 2 – Prompting Techniques Comparison")
    print("=" * 70)
    print("\nThree prompting methods are evaluated on the 5 Task 1 examples:")
    print("  1. zero_shot  – No in-context examples; baseline performance.")
    print("  2. few_shot   – 3 in-context demonstrations guide format/style.")
    print("  3. cot        – Step-by-step reasoning before the final answer.")

    all_results = []
    for model_name in MODELS:
        print(f"\n\n{'=' * 70}")
        print(f"Model: {model_name}")
        print("=" * 70)
        results = run_task2(model_name)
        all_results.extend(results)

    path = save_results(all_results, "task2_prompting_techniques", "both_models")
    print(f"\n[task2] Results saved to: {path}")


if __name__ == "__main__":
    main()