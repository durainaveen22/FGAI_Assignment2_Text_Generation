"""
evaluate.py - Quantitative and qualitative evaluation metrics, visualizations, and comparative analysis
Foundations of Generative AI (ICT4403) - Assignment 2
"""

import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def compute_distinct_ngrams(tokens, n=1):
    """
    Computes Distinct-n metric: unique n-grams / total n-grams.
    Higher values indicate greater lexical diversity.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)

def compute_repetition_rate(tokens):
    """
    Measures repetition rate: proportion of repeated consecutive tokens or n-grams.
    """
    if len(tokens) <= 1:
        return 0.0
    immediate_repeats = sum(1 for i in range(len(tokens) - 1) if tokens[i] == tokens[i + 1])
    return immediate_repeats / (len(tokens) - 1)

def evaluate_generated_samples(samples):
    """
    Evaluates qualitative characteristics across generated text samples:
    Distinct-1, Distinct-2, and Repetition Rate.
    """
    evaluated = []
    for item in samples:
        gen_tokens = item["generated_text"].lower().split()
        d1 = compute_distinct_ngrams(gen_tokens, n=1)
        d2 = compute_distinct_ngrams(gen_tokens, n=2)
        rep = compute_repetition_rate(gen_tokens)
        
        evaluated.append({
            **item,
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "repetition_rate": round(rep, 4),
            "num_words": len(gen_tokens)
        })
    return evaluated

def plot_loss_curves(histories, save_path):
    """
    Plots Training and Validation loss curves for all 4 models.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Simple_RNN": "#e74c3c", "BiRNN": "#e67e22", "LSTM": "#2980b9", "GRU": "#27ae60"}
    markers = {"Simple_RNN": "o", "BiRNN": "s", "LSTM": "^", "GRU": "d"}

    for name, hist in histories.items():
        epochs = hist["epochs"]
        c = colors.get(name, "black")
        m = markers.get(name, "o")
        ax1.plot(epochs, hist["train_loss"], label=f"{name}", color=c, marker=m, linewidth=2)
        ax2.plot(epochs, hist["val_loss"], label=f"{name}", color=c, marker=m, linewidth=2)

    ax1.set_title("Training Loss vs. Epochs", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Validation Loss vs. Epochs", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved loss curves plot to {save_path}")

def plot_perplexity_and_accuracy(histories, save_path):
    """
    Plots Validation Perplexity and Top-1 / Top-5 Accuracy comparisons.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(histories.keys())
    best_ppls = [histories[m]["best_val_ppl"] for m in model_names]
    top1_accs = [histories[m]["val_top1_acc"][-1] for m in model_names]
    top5_accs = [histories[m]["val_top5_acc"][-1] for m in model_names]

    bar_colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]

    # Perplexity bar chart
    bars1 = ax1.bar(model_names, best_ppls, color=bar_colors, width=0.55, edgecolor='black', alpha=0.85)
    ax1.set_title("Validation Perplexity (Lower is Better)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Perplexity (PPL)", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')

    # Accuracy grouped bar chart
    x = np.arange(len(model_names))
    width = 0.35
    b1 = ax2.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy (%)', color='#3498db', edgecolor='black')
    b2 = ax2.bar(x + width/2, top5_accs, width, label='Top-5 Accuracy (%)', color='#2ecc71', edgecolor='black')
    ax2.set_title("Validation Prediction Accuracy (%)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in b1:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)
    for bar in b2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved perplexity and accuracy plot to {save_path}")

def plot_training_time_and_params(histories, save_path):
    """
    Plots training duration and parameter counts across models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(histories.keys())
    times = [histories[m]["total_training_time"] for m in model_names]
    params = [histories[m]["parameters"] for m in model_names]

    colors = ["#e74c3c", "#e67e22", "#2980b9", "#27ae60"]

    bars1 = ax1.bar(model_names, times, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    ax1.set_title("Total Training Time (Seconds)", fontsize=13, fontweight='bold')
    ax1.set_ylabel("Seconds", fontsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}s", ha='center', va='bottom', fontweight='bold')

    bars2 = ax2.bar(model_names, [p / 1e3 for p in params], color=colors, width=0.55, edgecolor='black', alpha=0.85)
    ax2.set_title("Total Trainable Parameters (in Thousands)", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Parameters (K)", fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{yval:.1f}K", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training time and parameter plot to {save_path}")

def plot_temperature_diversity(evaluated_samples, save_path):
    """
    Plots the relationship between sampling temperature and lexical diversity (Distinct-1 & Distinct-2).
    """
    temps = sorted(list(set(s["temperature"] for s in evaluated_samples)))
    models = sorted(list(set(s["model"] for s in evaluated_samples)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Simple_RNN": "#e74c3c", "BiRNN": "#e67e22", "LSTM": "#2980b9", "GRU": "#27ae60"}

    for model in models:
        d1_means = []
        d2_means = []
        for t in temps:
            subset = [s for s in evaluated_samples if s["model"] == model and s["temperature"] == t]
            d1_means.append(np.mean([s["distinct_1"] for s in subset]))
            d2_means.append(np.mean([s["distinct_2"] for s in subset]))

        c = colors.get(model, "black")
        ax1.plot(temps, d1_means, label=model, marker="o", linewidth=2, color=c)
        ax2.plot(temps, d2_means, label=model, marker="s", linewidth=2, color=c)

    ax1.set_title("Distinct-1 (Unigram Diversity) vs Temperature", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Sampling Temperature (T)", fontsize=11)
    ax1.set_ylabel("Distinct-1 Ratio", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Distinct-2 (Bigram Diversity) vs Temperature", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Sampling Temperature (T)", fontsize=11)
    ax2.set_ylabel("Distinct-2 Ratio", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved temperature diversity plot to {save_path}")
