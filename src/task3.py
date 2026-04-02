"""Task 3 – Chain-of-thought (CoT) prompting experiments.

Evaluates both LLMs on all 30 dataset examples using a chain-of-thought prompt
that instructs the model to reason step-by-step before stating its answer.

Usage::

    python -m src.task3
    # or
    python src/task3.py

Results are written to results/task3_cot_<model_slug>.json.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_examples
from src.model_loader import MODELS, load_model, generate_text
from utils.prompt_templates import build_cot_prompt
from utils.evaluation import extract_answer, score_answer, save_results


EXPERIMENT = "task3_cot"


def run_cot(model_name: str) -> str:
    """Run chain-of-thought experiments for a single model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Path to the saved JSON results file.
    """
    examples = load_examples()
    model, tokenizer = load_model(model_name)

    results = []
    for example in examples:
        prompt = build_cot_prompt(example["question"])
        # Allow more tokens so the model can produce a reasoning chain
        raw_output = generate_text(prompt, model, tokenizer, max_new_tokens=150)
        predicted = extract_answer(raw_output, prompt_type="cot")
        score = score_answer(predicted, example["expected"])

        results.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected": example["expected"],
                "category": example["category"],
                "prompt": prompt,
                "output": raw_output,
                "predicted": predicted,
                "score": score,
            }
        )

        print(
            f"  [{example['id']:02d}] expected={example['expected']!r:20s} "
            f"predicted={predicted!r:20s}  score={score}"
        )

    output_path = save_results(results, EXPERIMENT, model_name)
    print(f"\n[task3] Results saved to: {output_path}")
    return output_path


def main() -> None:
    """Entry point: run CoT experiments for all models."""
    print("=" * 60)
    print("Task 3 – Chain-of-thought prompting")
    print("=" * 60)
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        run_cot(model_name)


if __name__ == "__main__":
    main()
