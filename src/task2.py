"""Task 2 – Few-shot prompting experiments.

Evaluates both LLMs using a 3-shot prompt.  The first 3 examples from the
dataset are used as demonstrations; the remaining 27 examples are evaluated.

Usage::

    python -m src.task2
    # or
    python src/task2.py

Results are written to results/task2_few_shot_<model_slug>.json.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_examples, get_few_shot_pool, get_evaluation_set
from src.model_loader import MODELS, load_model, generate_text
from utils.prompt_templates import build_few_shot_prompt
from utils.evaluation import extract_answer, score_answer, save_results


EXPERIMENT = "task2_few_shot"
N_SHOTS = 3


def run_few_shot(model_name: str) -> str:
    """Run few-shot experiments for a single model.

    The first *N_SHOTS* examples are used as demonstrations and are excluded
    from evaluation.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Path to the saved JSON results file.
    """
    examples = load_examples()
    demonstrations = get_few_shot_pool(examples, n=N_SHOTS)
    eval_examples = get_evaluation_set(examples, skip=N_SHOTS)

    model, tokenizer = load_model(model_name)

    results = []
    for example in eval_examples:
        prompt = build_few_shot_prompt(example["question"], demonstrations)
        raw_output = generate_text(prompt, model, tokenizer, max_new_tokens=60)
        predicted = extract_answer(raw_output, prompt_type="few_shot")
        score = score_answer(predicted, example["expected"])

        results.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected": example["expected"],
                "category": example["category"],
                "n_shots": N_SHOTS,
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
    print(f"\n[task2] Results saved to: {output_path}")
    return output_path


def main() -> None:
    """Entry point: run few-shot experiments for all models."""
    print("=" * 60)
    print(f"Task 2 – {N_SHOTS}-shot prompting")
    print("=" * 60)
    for model_name in MODELS:
        print(f"\n--- Model: {model_name} ---")
        run_few_shot(model_name)


if __name__ == "__main__":
    main()
