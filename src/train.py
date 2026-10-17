"""
train.py - Controlled training loop and evaluation across all 4 architectures
Foundations of Generative AI (ICT4403) - Assignment 2
"""

import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(logits, targets, topk=(1, 5)):
    """
    Computes top-k accuracy for next-word prediction.
    """
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=-1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item() / targets.size(0))
        return res

def train_epoch(model, dataloader, criterion, optimizer, device, clip_norm=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, targets in dataloader:
        sequences, targets = sequences.to(device), targets.to(device)
        optimizer.zero_grad()
        
        logits, _ = model(sequences)
        loss = criterion(logits, targets)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    epoch_loss = total_loss / total_samples
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0.0
    total_top5 = 0.0

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences, targets = sequences.to(device), targets.to(device)
            logits, _ = model(sequences)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size

    val_loss = total_loss / total_samples
    val_ppl = math.exp(min(val_loss, 20.0))  # guard against overflow
    val_top1 = total_top1 / total_samples
    val_top5 = total_top5 / total_samples
    return val_loss, val_ppl, val_top1, val_top5

def train_model(model_name, model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu', save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    param_count = model.count_parameters()
    print(f"\n==========================================")
    print(f"Training {model_name} ({param_count:,} parameters)")
    print(f"==========================================")

    history = {
        "model_name": model_name,
        "parameters": param_count,
        "epochs": [],
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "val_top1_acc": [],
        "val_top5_acc": [],
        "epoch_time": [],
        "total_training_time": 0.0,
        "best_val_loss": float("inf"),
        "best_val_ppl": float("inf")
    }

    start_total_time = time.time()
    best_loss = float("inf")
    best_weights_path = os.path.join(save_dir, f"{model_name.lower().replace(' ', '_')}_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_ppl = math.exp(min(train_loss, 20.0))
        
        val_loss, val_ppl, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)
        t_epoch = time.time() - t0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_ppl"].append(round(train_ppl, 2))
        history["val_loss"].append(round(val_loss, 4))
        history["val_ppl"].append(round(val_ppl, 2))
        history["val_top1_acc"].append(round(val_top1 * 100, 2))
        history["val_top5_acc"].append(round(val_top5 * 100, 2))
        history["epoch_time"].append(round(t_epoch, 2))

        if val_loss < best_loss:
            best_loss = val_loss
            history["best_val_loss"] = round(val_loss, 4)
            history["best_val_ppl"] = round(val_ppl, 2)
            torch.save(model.state_dict(), best_weights_path)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{t_epoch:.1f}s] - Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | Val Top-1: {val_top1*100:.2f}% | Val Top-5: {val_top5*100:.2f}%")

    total_time = time.time() - start_total_time
    history["total_training_time"] = round(total_time, 2)
    history["checkpoint_path"] = best_weights_path
    print(f"Finished {model_name} in {total_time:.2f}s | Best Val PPL: {history['best_val_ppl']}")

    return history
