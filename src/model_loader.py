"""Model loading utilities for the LLMs-and-Prompting project.

Supported models (1B-3B parameter range):
  - EleutherAI/gpt-neo-1.3B  (~1.3 B parameters)
  - EleutherAI/gpt-neo-2.7B  (~2.7 B parameters)

Both models are causal language models available on the HuggingFace Hub and
work with the standard ``transformers`` AutoModelForCausalLM / AutoTokenizer
API.
"""

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# The two models evaluated in all experiments
MODEL_1 = "EleutherAI/gpt-neo-1.3B"
MODEL_2 = "EleutherAI/gpt-neo-2.7B"

MODELS = [MODEL_1, MODEL_2]


def load_model(
    model_name: str,
    device: str = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokenizer from the HuggingFace Hub.

    The model is moved to *device* (auto-detected if not specified).
    ``torch_dtype=torch.float16`` is used when a CUDA device is available to
    reduce memory consumption.

    Args:
        model_name: HuggingFace model identifier (e.g. 'EleutherAI/gpt-neo-1.3B').
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

    # Some tokenizers (e.g. GPT-Neo) do not set a pad token by default.
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


def generate_text(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Generate text from a prompt using a causal language model.

    Greedy decoding is used by default (``temperature=0.0`` / ``do_sample=False``)
    for reproducibility.

    Args:
        prompt: The input prompt string.
        model: A loaded causal LM.
        tokenizer: The corresponding tokenizer.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.  Set to 0.0 for greedy decoding.

    Returns:
        The newly generated text (the prompt prefix is stripped).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
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

    # Decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0][input_length:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)
    return generated.strip()
