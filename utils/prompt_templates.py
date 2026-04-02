"""Prompt template builders for zero-shot, few-shot, and chain-of-thought prompting."""

from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(question: str) -> str:
    """Return a plain zero-shot prompt for the given question.

    Args:
        question: The question to answer.

    Returns:
        A formatted prompt string.
    """
    return (
        "Answer the following question concisely.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

def build_few_shot_prompt(
    question: str,
    demonstrations: List[Dict[str, Any]],
) -> str:
    """Return a few-shot prompt using the provided demonstrations.

    Each demonstration dict must have 'question' and 'expected' keys.

    Args:
        question: The question to answer.
        demonstrations: List of example dicts used as in-context demonstrations.

    Returns:
        A formatted prompt string.
    """
    lines = ["Answer each question concisely, as shown in the examples below.\n"]
    for demo in demonstrations:
        lines.append(f"Question: {demo['question']}")
        lines.append(f"Answer: {demo['expected']}\n")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chain-of-thought
# ---------------------------------------------------------------------------

_COT_DEMOS = [
    {
        "question": "What is 15 + 27?",
        "reasoning": (
            "I need to add 15 and 27. "
            "15 + 20 = 35, then 35 + 7 = 42."
        ),
        "answer": "42",
    },
    {
        "question": "A baker makes 24 cookies and sells them in packs of 6. How many packs can he make?",
        "reasoning": (
            "I need to divide 24 by 6. "
            "24 / 6 = 4."
        ),
        "answer": "4",
    },
    {
        "question": "What is the capital of France?",
        "reasoning": (
            "France is a country in Western Europe. "
            "Its capital city, where the government is located, is Paris."
        ),
        "answer": "Paris",
    },
]


def build_cot_prompt(question: str) -> str:
    """Return a chain-of-thought prompt with fixed few-shot CoT demonstrations.

    The prompt instructs the model to reason step-by-step before giving its
    final answer.

    Args:
        question: The question to answer.

    Returns:
        A formatted prompt string.
    """
    lines = [
        "Answer each question by first reasoning step-by-step, "
        "then state the final answer on a line that begins with 'Answer:'.\n"
    ]
    for demo in _COT_DEMOS:
        lines.append(f"Question: {demo['question']}")
        lines.append(f"Reasoning: {demo['reasoning']}")
        lines.append(f"Answer: {demo['answer']}\n")

    lines.append(f"Question: {question}")
    lines.append("Reasoning:")
    return "\n".join(lines)
