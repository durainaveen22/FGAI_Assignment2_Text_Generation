"""
main.py - End-to-end execution script for ICT4403 Assignment 2
Trains Simple RNN, Bidirectional RNN, LSTM, and GRU models for autoregressive text generation,
performs quantitative & qualitative evaluations, and runs temperature sampling studies.
"""

import os
import sys
import json
import argparse
import torch

# Ensure local src directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders
from src.models import get_all_models
from src.train import set_seed, train_model
from src.generate import run_temperature_experiment
from src.evaluate import (
    evaluate_generated_samples,
    plot_loss_curves,
    plot_perplexity_and_accuracy,
    plot_training_time_and_params,
    plot_temperature_diversity
)

def parse_args():
    parser = argparse.ArgumentParser(description="Foundations of Generative AI - Assignment 2")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs per model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=20, help="Input sequence length (sliding window)")
    parser.add_argument("--stride", type=int, default=3, help="Sliding window stride")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden state dimension")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    saved_models_dir = os.path.join(base_dir, "saved_models")
    results_dir = os.path.join(base_dir, "results")
    plots_dir = os.path.join(base_dir, "plots")

    for d in [saved_models_dir, results_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    print("=================================================================")
    print(" ICT4403: Foundations of Generative AI - Assignment 2")
    print(" Comparative Autoregressive Text Generation System")
    print(" Architectures: Simple RNN | Bidirectional RNN | LSTM | GRU")
    print(f" Execution Device: {args.device.upper()} | Seed: {args.seed}")
    print("=================================================================\n")

    # Step 1: Dataset Preparation
    print("[1/4] Preparing dataset and sequence dataloaders...")
    train_loader, val_loader, vocab, meta = get_dataloaders(
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        data_dir=data_dir
    )
    vocab_size = len(vocab)
    print(f"-> Vocabulary Size: {vocab_size} | Training Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # Step 2: Initialize Models
    print("\n[2/4] Initializing all 4 recurrent architectures with identical parameters...")
    models = get_all_models(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=1,
        dropout=0.1
    )

    for name, model in models.items():
        print(f" - {name:12s}: {model.count_parameters():,} trainable parameters")

    # Step 3: Train all models
    print("\n[3/4] Starting training loop under comparable experimental conditions...")
    histories = {}
    for name, model in models.items():
        hist = train_model(
            model_name=name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            save_dir=saved_models_dir
        )
        histories[name] = hist

    # Save training histories
    hist_path = os.path.join(results_dir, "training_histories.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    print(f"\nSaved training histories to {hist_path}")

    # Generate plots
    print("Generating evaluation plots...")
    plot_loss_curves(histories, os.path.join(plots_dir, "loss_curves.png"))
    plot_perplexity_and_accuracy(histories, os.path.join(plots_dir, "perplexity_and_accuracy.png"))
    plot_training_time_and_params(histories, os.path.join(plots_dir, "training_time_and_params.png"))

    # Step 4: Text Generation and Temperature Experiment
    print("\n[4/4] Conducting Autoregressive Text Generation & Temperature Sampling Study...")
    seed_prompts = [
        "the detective looked carefully at the",
        "it was a cold and dark night in",
        "holmes smiled and took his"
    ]
    temperatures = [0.3, 0.7, 1.2]

    # Load best weights into models for generation
    for name, model in models.items():
        best_pt = os.path.join(saved_models_dir, f"{name.lower()}_best.pt")
        if os.path.exists(best_pt):
            model.load_state_dict(torch.load(best_pt, map_location=args.device))
        model.eval()

    raw_generations = run_temperature_experiment(
        models_dict=models,
        vocab=vocab,
        seed_texts=seed_prompts,
        temperatures=temperatures,
        max_new_tokens=25,
        device=args.device
    )

    evaluated_generations = evaluate_generated_samples(raw_generations)
    gen_path = os.path.join(results_dir, "generation_results.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(evaluated_generations, f, indent=2)
    print(f"Saved {len(evaluated_generations)} generation samples to {gen_path}")

    # Plot temperature vs diversity
    plot_temperature_diversity(evaluated_generations, os.path.join(plots_dir, "diversity_vs_temperature.png"))

    # Print summary comparison table
    print("\n" + "="*85)
    print(f"{'Model':<14} | {'Params':<8} | {'Val Loss':<9} | {'Val PPL':<8} | {'Top-1 Acc':<10} | {'Top-5 Acc':<10} | {'Train Time':<10}")
    print("="*85)
    for name, hist in histories.items():
        print(f"{name:<14} | {hist['parameters']:<8,d} | {hist['best_val_loss']:<9.4f} | {hist['best_val_ppl']:<8.2f} | {hist['val_top1_acc'][-1]:<9.2f}% | {hist['val_top5_acc'][-1]:<9.2f}% | {hist['total_training_time']:<9.1f}s")
    print("="*85)

    print("\nSample Generated Texts (Seed: 'the detective looked carefully at the'):")
    print("-"*85)
    for item in evaluated_generations:
        if item["seed"] == "the detective looked carefully at the" and item["temperature"] == 0.7:
            print(f"[{item['model']:<12} | T=0.7]: {item['full_text']}")
    print("-"*85)

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
