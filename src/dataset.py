"""
dataset.py - Dataset preparation, preprocessing, and sequence generation
Foundations of Generative AI (ICT4403) - Assignment 2
"""

import os
import re
import json
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
EOS_IDX = 2

class Vocabulary:
    def __init__(self, min_freq=3):
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, EOS_TOKEN: EOS_IDX}
        self.idx2word = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN, EOS_IDX: EOS_TOKEN}
        self.word_counts = {}

    def build_vocab(self, tokens):
        for token in tokens:
            self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        idx = len(self.word2idx)
        for token, count in sorted(self.word_counts.items(), key=lambda x: -x[1]):
            if count >= self.min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens):
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices):
        return [self.idx2word.get(i, UNK_TOKEN) for i in indices]

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "word2idx": self.word2idx,
                "min_freq": self.min_freq
            }, f, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab


class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def download_dataset(target_dir=DATA_DIR):
    os.makedirs(target_dir, exist_ok=True)
    raw_path = os.path.join(target_dir, "sherlock_raw.txt")
    
    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 10000:
        print(f"Dataset already exists at {raw_path}")
        return raw_path

    print(f"Downloading Adventures of Sherlock Holmes from {GUTENBERG_URL} ...")
    req = urllib.request.Request(
        GUTENBERG_URL,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    with urllib.request.urlopen(req) as response:
        content = response.read().decode('utf-8', errors='ignore')
    
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Downloaded {len(content)} characters to {raw_path}")
    return raw_path


def clean_text(text):
    """
    Strips Gutenberg headers/footers and normalizes text.
    """
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)

    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]
    
    text = re.sub(r"\[Illustration[^\]]*\]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("--", " - ")
    text = text.lower().strip()
    return text


def tokenize(text):
    """
    Tokenizes text while treating punctuation marks as separate tokens for coherent language modeling.
    """
    text = re.sub(r"([.,!?;:\(\)\"\'])", r" \1 ", text)
    tokens = text.split()
    return tokens


def prepare_sequences(tokens, vocab, seq_len=20, stride=3):
    """
    Constructs input-target pairs using a sliding window.
    Input: [w_1, w_2, ..., w_L]
    Target: w_{L+1}
    """
    encoded = vocab.encode(tokens)
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len, stride):
        inputs.append(encoded[i : i + seq_len])
        targets.append(encoded[i + seq_len])
    return inputs, targets


def get_dataloaders(seq_len=20, stride=3, batch_size=64, train_ratio=0.8, min_freq=3, data_dir=DATA_DIR):
    """
    End-to-end dataset pipeline returning DataLoaders, Vocabulary, and corpus statistics.
    """
    raw_path = download_dataset(data_dir)
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)
    clean_path = os.path.join(data_dir, "sherlock_clean.txt")
    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    tokens = tokenize(cleaned)
    print(f"Total tokens in corpus: {len(tokens)}")

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(tokens)
    print(f"Vocabulary size: {len(vocab)} (min_freq={min_freq})")
    vocab.save(os.path.join(data_dir, "vocab.json"))

    inputs, targets = prepare_sequences(tokens, vocab, seq_len=seq_len, stride=stride)
    total_samples = len(inputs)
    print(f"Constructed {total_samples} input-target sequence pairs (seq_len={seq_len}, stride={stride})")

    split_idx = int(total_samples * train_ratio)
    train_inputs, train_targets = inputs[:split_idx], targets[:split_idx]
    val_inputs, val_targets = inputs[split_idx:], targets[split_idx:]

    print(f"Train samples: {len(train_inputs)}, Validation samples: {len(val_inputs)}")

    train_dataset = TextDataset(train_inputs, train_targets)
    val_dataset = TextDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    metadata = {
        "dataset_name": "The Adventures of Sherlock Holmes (Arthur Conan Doyle)",
        "source_url": GUTENBERG_URL,
        "raw_characters": len(raw_text),
        "total_tokens": len(tokens),
        "vocab_size": len(vocab),
        "seq_len": seq_len,
        "stride": stride,
        "total_sequences": total_samples,
        "train_samples": len(train_inputs),
        "val_samples": len(val_inputs),
        "min_freq": min_freq
    }

    with open(os.path.join(data_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return train_loader, val_loader, vocab, metadata


if __name__ == "__main__":
    t_loader, v_loader, vocab, meta = get_dataloaders()
    for seq, tgt in t_loader:
        print("Sample batch input shape:", seq.shape)
        print("Sample batch target shape:", tgt.shape)
        sample_words = vocab.decode(seq[0].tolist())
        target_word = vocab.decode([tgt[0].item()])[0]
        print("Sample input:", " ".join(sample_words))
        print("Sample target:", target_word)
        break
