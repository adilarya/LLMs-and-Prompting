"""Model loading utilities for the LLMs-and-Prompting project.

Supported models (1B-3B parameter range, instruction-tuned):
  - HuggingFaceTB/SmolLM3-3B  (~3 B parameters, HuggingFace SmolLM family)
  - Qwen/Qwen2.5-3B-Instruct  (~3.09 B parameters, Alibaba Qwen family)

Both are instruction-tuned chat models that use the standard HuggingFace
``AutoModelForCausalLM`` / ``AutoTokenizer`` API with ``apply_chat_template``
for proper message formatting.
"""

from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# The two models evaluated in all experiments.
# They are similar in size (~3 B) but come from different model families.
MODEL_1 = "HuggingFaceTB/SmolLM3-3B"
MODEL_2 = "Qwen/Qwen2.5-3B-Instruct"

MODELS = [MODEL_1, MODEL_2]


def short_name(model_name: str) -> str:
    """Return the final path component of a HuggingFace model identifier.

    Args:
        model_name: Full HuggingFace model identifier (e.g. 'HuggingFaceTB/SmolLM3-3B').

    Returns:
        The part after the last '/' (e.g. 'SmolLM3-3B').
    """
    return model_name.split("/")[-1]


def load_model(
    model_name: str,
    device: str = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load an instruction-tuned causal LM and its tokenizer from the HuggingFace Hub.

    The model is moved to *device* (auto-detected if not specified).
    ``torch_dtype=torch.float16`` is used when a CUDA device is available to
    reduce memory consumption.

    Args:
        model_name: HuggingFace model identifier
                    (e.g. 'HuggingFaceTB/SmolLM3-3B').
        device: Target device string ('cuda', 'cpu', …).  If *None* the
                function selects CUDA when available, otherwise CPU.

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[model_loader] Loading '{model_name}' on {device} …")

    use_fp16 = device.startswith("cuda")
    dtype = torch.float16 if use_fp16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    print(f"[model_loader] '{model_name}' ready.")
    return model, tokenizer


def generate_chat(
    messages: List[Dict[str, str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    seed: int = 42,
) -> str:
    """Generate a response for a list of chat messages using an instruction-tuned model.

    The messages are formatted via the tokenizer's ``apply_chat_template``
    method so that model-specific special tokens are inserted correctly.
    Greedy decoding (``temperature=0.0``) is the default for reproducibility.

    Args:
        messages: List of dicts with 'role' ('system'/'user'/'assistant') and
                  'content' keys.
        model: A loaded causal LM.
        tokenizer: The corresponding tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.  Set to 0.0 for greedy decoding.
        seed: Random seed (used only when ``temperature > 0``).

    Returns:
        The newly generated assistant text (prompt prefix is stripped).
    """
    if temperature > 0.0:
        torch.manual_seed(seed)

    device = next(model.parameters()).device

    # Format the conversation using the model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = output_ids[0][input_length:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated.strip()


def generate_text(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Generate text from a raw prompt string (legacy helper).

    Wraps *prompt* as a single user message and delegates to
    :func:`generate_chat`.

    Args:
        prompt: The input prompt string.
        model: A loaded causal LM.
        tokenizer: The corresponding tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.  Set to 0.0 for greedy decoding.

    Returns:
        The newly generated text (the prompt prefix is stripped).
    """
    messages = [{"role": "user", "content": prompt}]
    return generate_chat(messages, model, tokenizer, max_new_tokens, temperature)
