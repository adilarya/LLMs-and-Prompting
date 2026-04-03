"""Task 1 – Model Selection and Original Example Comparison.

Runs five original (prompt, expected answer) examples on both models using
zero-shot prompting, then prints a side-by-side comparison table and saves
the results to JSON.

The five examples are designed for Task 3C (Constrained Decoding and
Structured Output): they test whether the models can respond with a specific
label, category, or concise structured answer.

Usage::

    python -m src.task1
    # or
    python src/task1.py

Results are written to results/task1_model_comparison.json.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import MODELS, MODEL_1, MODEL_2, load_model, generate_chat, short_name
from utils.prompt_templates import build_zero_shot_messages
from utils.evaluation import extract_answer, score_answer, save_results


# ---------------------------------------------------------------------------
# Five original examples (Task 1)
# Each has a clearly defined expected answer or acceptable answer set.
# ---------------------------------------------------------------------------

TASK1_EXAMPLES = [
    {
        "id": 1,
        "task": "Sentiment classification (constrained label)",
        "question": (
            "Classify the sentiment of the following review as exactly one of: "
            "Positive, Negative, or Neutral.\n\n"
            "Review: \"The battery died after two hours and customer support never replied.\""
        ),
        "expected": "Negative",
        "acceptable": {"negative"},
        "category": "classification",
    },
    {
        "id": 2,
        "task": "Multiple-choice factual question",
        "question": (
            "Answer with only the letter of the correct option.\n\n"
            "Which of the following is a prime number?\n"
            "A) 4\nB) 6\nC) 11\nD) 15"
        ),
        "expected": "C",
        "acceptable": {"c", "c) 11", "11"},
        "category": "multiple_choice",
    },
    {
        "id": 3,
        "task": "Structured JSON extraction",
        "question": (
            "Extract the name and age from the sentence below and return them "
            "in JSON format with keys \"name\" and \"age\".\n\n"
            "Sentence: \"Maria just turned 31 last Tuesday.\""
        ),
        "expected": '{"name": "Maria", "age": 31}',
        "acceptable": {"maria", '"age": 31', "age\": 31"},
        "category": "json_extraction",
    },
    {
        "id": 4,
        "task": "Country identification from clue",
        "question": (
            "Name the country described by the clue below. "
            "Give only the country name.\n\n"
            "Clue: \"This is the world's largest country by land area and spans eleven time zones.\""
        ),
        "expected": "Russia",
        "acceptable": {"russia"},
        "category": "factual_constrained",
    },
    {
        "id": 5,
        "task": "Part-of-speech label",
        "question": (
            "Identify the part of speech of the underlined word. "
            "Choose one: noun, verb, adjective, adverb, preposition.\n\n"
            "Sentence: \"She ran **quickly** to catch the bus.\"\n"
            "Underlined word: quickly"
        ),
        "expected": "adverb",
        "acceptable": {"adverb"},
        "category": "linguistic_label",
    },
]


def _is_acceptable(raw_output: str, acceptable: set) -> int:
    """Return 1 if the raw output contains any acceptable answer (case-insensitive)."""
    low = raw_output.lower()
    return int(any(ans in low for ans in acceptable))


def run_task1(model_name: str) -> list:
    """Run Task 1 zero-shot comparison for a single model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        List of result dicts for each example.
    """
    model, tokenizer = load_model(model_name)

    results = []
    for ex in TASK1_EXAMPLES:
        messages = build_zero_shot_messages(ex["question"])
        raw_output = generate_chat(messages, model, tokenizer, max_new_tokens=80)
        correct = _is_acceptable(raw_output, ex["acceptable"])

        results.append(
            {
                "id": ex["id"],
                "task": ex["task"],
                "question": ex["question"],
                "expected": ex["expected"],
                "category": ex["category"],
                "model": model_name,
                "output": raw_output,
                "correct": correct,
            }
        )
    return results


def main() -> None:
    """Entry point: run Task 1 on both models and print comparison table."""
    print("=" * 70)
    print("Task 1 – Model Selection & Original Example Comparison")
    print("=" * 70)

    all_results: dict = {}
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        results = run_task1(model_name)
        all_results[model_name] = results

    # Print side-by-side comparison
    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    for ex in TASK1_EXAMPLES:
        idx = ex["id"] - 1
        r1 = all_results[MODEL_1][idx]
        r2 = all_results[MODEL_2][idx]
        print(f"\nExample {ex['id']}: {ex['task']}")
        print(f"  Expected : {ex['expected']}")
        print(f"  {short_name(MODEL_1):40s}: {r1['output'][:120]!r}  [{'OK' if r1['correct'] else 'WRONG'}]")
        print(f"  {short_name(MODEL_2):40s}: {r2['output'][:120]!r}  [{'OK' if r2['correct'] else 'WRONG'}]")

    # Save combined results
    combined = []
    for model_name in MODELS:
        combined.extend(all_results[model_name])
    path = save_results(combined, "task1_model_comparison", "both_models")
    print(f"\n[task1] Results saved to: {path}")


if __name__ == "__main__":
    main()
