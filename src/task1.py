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
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_loader import MODELS, MODEL_1, MODEL_2, load_model, generate_chat, short_name
from utils.prompt_templates import build_zero_shot_messages
from utils.evaluation import extract_answer, save_results


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
        "category": "linguistic_label",
    },
]


def _normalize(text: str) -> str:
    return text.strip().lower()


def _extract_multiple_choice_letter(text: str) -> str:
    """Extract a single answer letter like A/B/C/D from model text."""
    text = text.strip()

    # exact single-letter answer
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()

    # patterns like "C)", "C.", "C:"
    m = re.match(r"^\s*([A-Da-d])[\)\.\:\-]?\s*$", text)
    if m:
        return m.group(1).upper()

    # first standalone option letter
    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()

    return text.strip().upper()


def _extract_json_object(text: str):
    """Try to extract the first JSON object from text."""
    text = text.strip()

    # direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # try to find {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def evaluate_example(example: dict, raw_output: str) -> tuple[str, int, str]:
    """Return (predicted, correct, note) for one example."""
    category = example["category"]

    if category in {"classification", "factual_constrained", "linguistic_label"}:
        predicted = extract_answer(raw_output)
        correct = int(_normalize(predicted) == _normalize(example["expected"]))
        note = (
            "Correct exact label."
            if correct
            else f"Incorrect label. Expected {example['expected']}."
        )
        return predicted, correct, note

    if category == "multiple_choice":
        predicted = _extract_multiple_choice_letter(extract_answer(raw_output))
        correct = int(predicted == example["expected"].upper())
        if correct:
            note = "Correct option letter."
        else:
            note = f"Wrong option format or answer. Expected only {example['expected']}."
        return predicted, correct, note

    if category == "json_extraction":
        obj = _extract_json_object(raw_output)
        if obj is None:
            return raw_output.strip(), 0, "Invalid JSON."

        predicted = json.dumps(obj, ensure_ascii=False)
        correct = int(
            isinstance(obj, dict)
            and obj.get("name") == "Maria"
            and obj.get("age") == 31
        )
        note = (
            "Valid JSON with correct fields."
            if correct
            else "JSON parsed, but fields/values are incorrect."
        )
        return predicted, correct, note

    predicted = extract_answer(raw_output)
    correct = 0
    note = "Unknown category."
    return predicted, correct, note


def run_task1(model_name: str) -> list:
    """Run Task 1 zero-shot comparison for a single model."""
    model, tokenizer = load_model(model_name)

    results = []
    for ex in TASK1_EXAMPLES:
        messages = build_zero_shot_messages(ex["question"])
        raw_output = generate_chat(messages, model, tokenizer, max_new_tokens=80)

        predicted, correct, note = evaluate_example(ex, raw_output)

        results.append(
            {
                "id": ex["id"],
                "task": ex["task"],
                "question": ex["question"],
                "expected": ex["expected"],
                "category": ex["category"],
                "model": model_name,
                "raw_output": raw_output,
                "predicted": predicted,
                "correct": correct,
                "note": note,
            }
        )
    return results


def main() -> None:
    """Entry point: run Task 1 on both models and print comparison table."""
    print("=" * 70)
    print("Task 1 – Model Selection & Original Example Comparison")
    print("=" * 70)

    all_results = {}
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        results = run_task1(model_name)
        all_results[model_name] = results

    print("\n\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    for ex in TASK1_EXAMPLES:
        idx = ex["id"] - 1
        r1 = all_results[MODEL_1][idx]
        r2 = all_results[MODEL_2][idx]

        print(f"\nExample {ex['id']}: {ex['task']}")
        print(f"  Prompt    : {ex['question'][:100]}...")
        print(f"  Expected  : {ex['expected']}")
        print(
            f"  {short_name(MODEL_1):40s}: "
            f"{r1['predicted'][:80]!r}  "
            f"[{'OK' if r1['correct'] else 'WRONG'}]  "
            f"{r1['note']}"
        )
        print(
            f"  {short_name(MODEL_2):40s}: "
            f"{r2['predicted'][:80]!r}  "
            f"[{'OK' if r2['correct'] else 'WRONG'}]  "
            f"{r2['note']}"
        )

    combined = []
    for model_name in MODELS:
        combined.extend(all_results[model_name])

    path = save_results(combined, "task1_model_comparison", "both_models")
    print(f"\n[task1] Results saved to: {path}")


if __name__ == "__main__":
    main()