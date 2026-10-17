"""
models.py - Four Recurrent Neural Network Architectures for Autoregressive Text Generation
Simple RNN, Bidirectional RNN, LSTM, and GRU
Foundations of Generative AI (ICT4403) - Assignment 2
"""

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleRNNModel(BaseModel):
    """
    Standard Elman RNN for autoregressive next-word prediction.
    Hidden update: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.model_name = "Simple RNN"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(embeds, hidden)  # out: (batch_size, seq_len, hidden_dim)
        last_out = out[:, -1, :]  # (batch_size, hidden_dim)
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)  # (batch_size, vocab_size)
        return logits, hidden


class BiRNNModel(BaseModel):
    """
    Bidirectional RNN for prefix sequence encoding in next-word prediction.
    Processes the prefix sequence forward (w_1 -> w_t) and backward (w_t <- w_1).
    The combined representation at the terminal timestep encapsulates bidirectional prefix context.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.model_name = "Bidirectional RNN"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(embeds, hidden)  # out: (batch_size, seq_len, hidden_dim * 2)
        last_out = out[:, -1, :]  # (batch_size, hidden_dim * 2)
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)  # (batch_size, vocab_size)
        return logits, hidden


class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) network with input, forget, and output gating mechanisms
    and an internal cell memory state c_t to mitigate vanishing gradients.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.model_name = "LSTM"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)  # out: (batch_size, seq_len, hidden_dim)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits, hidden


class GRUModel(BaseModel):
    """
    Gated Recurrent Unit (GRU) with reset (r_t) and update (z_t) gates.
    Maintains expressive memory control with fewer parameters than LSTM.
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.model_name = "GRU"
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embeds = self.embedding(x)
        out, hidden = self.gru(embeds, hidden)  # out: (batch_size, seq_len, hidden_dim)
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        logits = self.fc(last_out)
        return logits, hidden


def get_all_models(vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1):
    """
    Instantiates all 4 models under identical hyperparameter conditions.
    """
    models = {
        "Simple_RNN": SimpleRNNModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "BiRNN": BiRNNModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout),
        "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    }
    return models
