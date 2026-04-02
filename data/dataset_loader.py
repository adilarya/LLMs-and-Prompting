"""Dataset loading utilities for the LLMs-and-Prompting project."""

import json
import os
from typing import List, Dict, Any


def load_examples(path: str = None) -> List[Dict[str, Any]]:
    """Load the 30-example dataset from the local JSON file.

    Args:
        path: Optional path to the JSON file. Defaults to data/examples.json
              relative to the project root.

    Returns:
        List of example dicts with keys: id, question, expected, category.
    """
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "data", "examples.json")

    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    return examples


def get_few_shot_pool(examples: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """Return the first *n* examples to use as a fixed few-shot demonstration pool.

    These examples are taken from the beginning of the dataset so that
    experiments are reproducible.  They should *not* be included when
    evaluating the remaining examples.

    Args:
        examples: Full list of dataset examples.
        n: Number of demonstrations to return.

    Returns:
        List of *n* examples from the dataset.
    """
    return examples[:n]


def get_evaluation_set(examples: List[Dict[str, Any]], skip: int = 0) -> List[Dict[str, Any]]:
    """Return the examples used for evaluation (everything after the few-shot pool).

    Args:
        examples: Full list of dataset examples.
        skip: Number of examples to skip at the start (e.g. the few-shot pool).
              Defaults to 0 so the full 30 examples are evaluated for zero-shot
              and CoT tasks that do not require a held-out demo pool.

    Returns:
        Subset of examples intended for evaluation.
    """
    return examples[skip:]
