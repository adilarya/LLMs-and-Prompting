"""Prompt template builders for zero-shot, few-shot, and chain-of-thought prompting.

Each builder has two forms:
  - ``build_*_messages(...)``  → returns a list of chat-role dicts suitable for
    ``tokenizer.apply_chat_template()``.  Use these with instruction-tuned models.
  - ``build_*_prompt(...)``    → returns a plain string (legacy, kept for
    backward compatibility).
"""

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = "You are a helpful assistant. Answer questions concisely."


def build_zero_shot_messages(question: str) -> List[Dict[str, str]]:
    """Return zero-shot chat messages for an instruction-tuned model.

    Args:
        question: The question to answer.

    Returns:
        List of role/content dicts.
    """
    return [
        {"role": "system", "content": _ZERO_SHOT_SYSTEM},
        {"role": "user", "content": f"Answer the following question concisely.\n\nQuestion: {question}"},
    ]


def build_zero_shot_prompt(question: str) -> str:
    """Return a plain zero-shot prompt string (legacy).

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

_FEW_SHOT_SYSTEM = (
    "You are a helpful assistant. "
    "Answer each question concisely, exactly as shown in the examples."
)


def build_few_shot_messages(
    question: str,
    demonstrations: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return few-shot chat messages for an instruction-tuned model.

    The demonstrations are embedded in the user turn as in-context examples.
    Each demonstration dict must have 'question' and 'expected' keys.

    Args:
        question: The question to answer.
        demonstrations: List of example dicts used as in-context demonstrations.

    Returns:
        List of role/content dicts.
    """
    demo_lines = ["Here are some examples:\n"]
    for demo in demonstrations:
        demo_lines.append(f"Question: {demo['question']}")
        demo_lines.append(f"Answer: {demo['expected']}\n")
    demo_lines.append("Now answer the following question concisely.")
    demo_lines.append(f"\nQuestion: {question}")
    user_content = "\n".join(demo_lines)
    return [
        {"role": "system", "content": _FEW_SHOT_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_few_shot_prompt(
    question: str,
    demonstrations: List[Dict[str, Any]],
) -> str:
    """Return a few-shot prompt string (legacy).

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

_COT_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Think step-by-step before giving your final answer. "
    "Always end your response with 'Answer: <your answer>'."
)

_COT_DEMOS = [
    {
        "question": "What is 15 + 27?",
        "reasoning": "I need to add 15 and 27. 15 + 20 = 35, then 35 + 7 = 42.",
        "answer": "42",
    },
    {
        "question": "A baker makes 24 cookies and sells them in packs of 6. How many packs can he make?",
        "reasoning": "I need to divide 24 by 6. 24 / 6 = 4.",
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


def build_cot_messages(question: str) -> List[Dict[str, str]]:
    """Return chain-of-thought chat messages for an instruction-tuned model.

    Fixed few-shot CoT demonstrations are embedded in the user turn.

    Args:
        question: The question to answer.

    Returns:
        List of role/content dicts.
    """
    demo_lines = ["Here are examples of step-by-step reasoning:\n"]
    for demo in _COT_DEMOS:
        demo_lines.append(f"Question: {demo['question']}")
        demo_lines.append(f"Reasoning: {demo['reasoning']}")
        demo_lines.append(f"Answer: {demo['answer']}\n")
    demo_lines.append(
        "Now solve the following question step-by-step. "
        "End your response with 'Answer: <your answer>'."
    )
    demo_lines.append(f"\nQuestion: {question}")
    user_content = "\n".join(demo_lines)
    return [
        {"role": "system", "content": _COT_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def build_cot_prompt(question: str) -> str:
    """Return a chain-of-thought prompt string (legacy).

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
