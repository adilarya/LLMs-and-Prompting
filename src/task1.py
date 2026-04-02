"""Task 1 – Zero-shot prompting experiments.

Evaluates both LLMs (EleutherAI/gpt-neo-1.3B and EleutherAI/gpt-neo-2.7B)
on all 30 dataset examples using a simple zero-shot prompt.

Usage::

    python -m src.task1
    # or
    python src/task1.py

Results are written to results/task1_zero_shot_<model_slug>.json.
"""

import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_examples
from src.model_loader import MODELS, load_model, generate_text
from utils.prompt_templates import build_zero_shot_prompt
from utils.evaluation import extract_answer, score_answer, save_results


EXPERIMENT = "task1_zero_shot"


def run_zero_shot(model_name: str) -> str:
    """Run zero-shot experiments for a single model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Path to the saved JSON results file.
    """
    examples = load_examples()
    model, tokenizer = load_model(model_name)

    results = []
    for example in examples:
        prompt = build_zero_shot_prompt(example["question"])
        raw_output = generate_text(prompt, model, tokenizer, max_new_tokens=60)
        predicted = extract_answer(raw_output, prompt_type="zero_shot")
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
    print(f"\n[task1] Results saved to: {output_path}")
    return output_path


def main() -> None:
    """Entry point: run zero-shot experiments for all models."""
    print("=" * 60)
    print("Task 1 – Zero-shot prompting")
    print("=" * 60)
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        run_zero_shot(model_name)


if __name__ == "__main__":
    main()
