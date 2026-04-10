"""Task 3 – Dataset-Based Evaluation (Task 3C: Constrained Decoding and Structured Output)."""

import sys
import os
import re
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_examples
from src.model_loader import MODELS, load_model, generate_chat, short_name
from utils.prompt_templates import (
    build_zero_shot_messages,
    build_few_shot_messages,
    build_cot_messages,
)
from utils.evaluation import save_task3_json


OUTPUT_FILENAME = "csci5541-s26-hw5-adilarya-3c.json"
TASK_FAMILY = "3c"
DATASET_NAME = "custom_constrained_output_dataset"

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

# Separate demo pool: NOT part of the 30 evaluation examples
DEMO_POOL = [
    {
        "question": (
            "Classify the sentiment of the following review as exactly one of: "
            "Positive, Negative, or Neutral.\n\n"
            "Review: \"The packaging was damaged, but the product worked fine.\""
        ),
        "expected": "Neutral",
    },
    {
        "question": (
            "Answer with only the letter of the correct option.\n\n"
            "Which planet is known as the Red Planet?\n"
            "A) Venus\nB) Mars\nC) Jupiter\nD) Saturn"
        ),
        "expected": "B",
    },
    {
        "question": (
            "Extract the city and country from the sentence below and return them "
            "in JSON format with keys \"city\" and \"country\".\n\n"
            "Sentence: \"Kyoto is one of the most historic cities in Japan.\""
        ),
        "expected": '{"city": "Kyoto", "country": "Japan"}',
    },
]


def _build_messages(question: str, method: str) -> list:
    if method == "zero_shot":
        return build_zero_shot_messages(question)
    if method == "few_shot":
        return build_few_shot_messages(question, DEMO_POOL)
    return build_cot_messages(question)


def _normalize(text: str) -> str:
    return text.strip().lower()


def _extract_answer(raw_output: str) -> str:
    for line in raw_output.splitlines():
        line = line.strip()
        if line:
            return line
    return raw_output.strip()


def _extract_mc_letter(text: str) -> str:
    text = text.strip()

    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper()

    m = re.match(r"^\s*([A-Da-d])[\)\.\:\-]?\s*$", text)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-Da-d])\b", text)
    if m:
        return m.group(1).upper()

    return text.upper()


def _extract_json_object(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def evaluate_dataset_example(example: dict, raw_output: str):
    category = example["category"]

    if category in {"classification", "factual_constrained", "linguistic_label"}:
        predicted = _extract_answer(raw_output)
        exact = int(_normalize(predicted) == _normalize(example["expected"]))
        format_valid = int(bool(predicted.strip()))
        note = "Exact label match." if exact else "Incorrect constrained label."
        return predicted, exact, format_valid, note

    if category == "multiple_choice":
        predicted = _extract_mc_letter(_extract_answer(raw_output))
        exact = int(predicted == example["expected"].upper())
        format_valid = int(predicted in {"A", "B", "C", "D"})
        note = "Correct option letter." if exact else "Wrong MCQ answer or format."
        return predicted, exact, format_valid, note

    if category == "json_extraction":
        obj = _extract_json_object(raw_output)
        if obj is None:
            return raw_output.strip(), 0, 0, "Invalid JSON."

        predicted = json.dumps(obj, ensure_ascii=False, sort_keys=True)

        try:
            expected_obj = json.loads(example["expected"])
        except Exception:
            expected_obj = None

        exact = int(obj == expected_obj)
        format_valid = 1
        note = "Valid JSON." if exact else "JSON parsed but content differed."
        return predicted, exact, format_valid, note

    predicted = _extract_answer(raw_output)
    return predicted, 0, int(bool(predicted.strip())), "Unknown category."


def run_task3(model_name: str, examples: list) -> list:
    model, tokenizer = load_model(model_name)
    records = []

    for method in PROMPTING_METHODS:
        print(f"\n  [Model: {short_name(model_name)}  Method: {method}]")
        max_tokens = MAX_NEW_TOKENS[method]

        for example in examples:
            messages = _build_messages(example["question"], method)
            raw_output = generate_chat(
                messages,
                model,
                tokenizer,
                max_new_tokens=max_tokens,
            )

            predicted, exact, fmt_valid, note = evaluate_dataset_example(example, raw_output)

            print(
                f"    [{example['id']:02d}] expected={example['expected']!r:15s} "
                f"predicted={predicted!r:25s}  exact={exact}"
            )

            record = {
                "example_id": None,
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
                    "notes": f"category={example['category']}; predicted={predicted!r}; {note}",
                },
                "generation_config": {
                    **_GEN_CONFIG_BASE,
                    "max_new_tokens": max_tokens,
                },
            }
            records.append(record)

    return records


def main() -> None:
    print("=" * 70)
    print("Task 3 – Dataset-Based Evaluation (3C: Constrained Output)")
    print("=" * 70)

    examples = load_examples()

    print(f"Dataset : {DATASET_NAME}  ({len(examples)} examples)")
    print(f"Methods : {PROMPTING_METHODS}")
    print(f"Models  : {MODELS}")
    print(
        f"Expected outputs: {len(examples)} × {len(PROMPTING_METHODS)} × {len(MODELS)} = "
        f"{len(examples) * len(PROMPTING_METHODS) * len(MODELS)}"
    )

    all_records = []
    for model_name in MODELS:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print("=" * 70)
        records = run_task3(model_name, examples)
        all_records.extend(records)

    for idx, record in enumerate(all_records, start=1):
        record["example_id"] = idx

    filepath = save_task3_json(all_records, OUTPUT_FILENAME)
    print(f"\n[task3] {len(all_records)} records saved to: {filepath}")

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