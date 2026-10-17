"""
generate.py - Autoregressive text generation and temperature-controlled sampling
Foundations of Generative AI (ICT4403) - Assignment 2
"""

import re
import torch
import torch.nn.functional as F
from .dataset import tokenize, UNK_IDX, PAD_IDX

def detokenize(tokens):
    """
    Reconstructs clean, human-readable text from token list, attaching punctuation naturally.
    """
    text = " ".join(tokens)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:\)])', r'\1', text)
    # Remove space after opening parenthesis
    text = re.sub(r'(\()\s+', r'\1', text)
    # Fix quotes
    text = re.sub(r'"\s*(.*?)\s*"', r'"\1"', text)
    text = re.sub(r"'\s*(.*?)\s*'", r"'\1'", text)
    return text.strip()


def generate_next_word(model, input_tensor, temperature=1.0, top_k=None, repetition_penalty=1.0, past_tokens=None):
    """
    Generates the next token index given the current input sequence.
    """
    with torch.no_grad():
        logits, _ = model(input_tensor)
        # logits shape: (1, vocab_size)
        logits = logits.squeeze(0)

        # Apply repetition penalty if specified
        if repetition_penalty != 1.0 and past_tokens:
            for token_id in set(past_tokens):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Prevent generating PAD or UNK directly during inference
        logits[PAD_IDX] = -float('inf')
        logits[UNK_IDX] = -float('inf')

        # Greedy sampling for very low temperature
        if temperature < 1e-3:
            return torch.argmax(logits, dim=-1).item()

        # Scale by temperature
        logits = logits / temperature

        # Optional Top-K truncation
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))
            val, _ = torch.topk(logits, top_k)
            logits[logits < val[-1]] = -float('inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        return next_token


def generate_text(model, vocab, seed_text, max_new_tokens=30, temperature=1.0, top_k=None, repetition_penalty=1.0, device='cpu', seq_len=20):
    """
    Autoregressively generates text starting from a seed prompt.
    """
    model.eval()
    model = model.to(device)

    seed_tokens = tokenize(seed_text)
    if not seed_tokens:
        seed_tokens = ["the"]

    encoded = vocab.encode(seed_tokens)
    generated_indices = list(encoded)

    for _ in range(max_new_tokens):
        # Prepare context window of length seq_len
        current_seq = generated_indices[-seq_len:]
        if len(current_seq) < seq_len:
            # Left-pad if seed is shorter than seq_len
            current_seq = [PAD_IDX] * (seq_len - len(current_seq)) + current_seq

        input_tensor = torch.tensor([current_seq], dtype=torch.long, device=device)
        next_token = generate_next_word(
            model,
            input_tensor,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            past_tokens=generated_indices[-10:]
        )
        generated_indices.append(next_token)

    decoded_tokens = vocab.decode(generated_indices)
    full_text = detokenize(decoded_tokens)
    new_text = detokenize(vocab.decode(generated_indices[len(encoded):]))
    return full_text, new_text


def run_temperature_experiment(models_dict, vocab, seed_texts, temperatures=[0.3, 0.7, 1.2], max_new_tokens=25, device='cpu'):
    """
    Generates text across models, seeds, and temperature conditions for qualitative/quantitative study.
    """
    results = []

    for seed in seed_texts:
        for temp in temperatures:
            for model_name, model in models_dict.items():
                full_text, new_text = generate_text(
                    model=model,
                    vocab=vocab,
                    seed_text=seed,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    device=device
                )
                results.append({
                    "model": model_name,
                    "seed": seed,
                    "temperature": temp,
                    "full_text": full_text,
                    "generated_text": new_text
                })
    return results
