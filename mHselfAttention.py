import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, block_size, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        p = query @ key.transpose(-2, -1) * C ** -0.5  # (B, T, T),每个元素再除以C的平方根，为了缩放
        p = F.softmax(p, dim=-1)
        p = p.masked_fill(self.tril == 0, float('-inf'))
        p = self.dropout(p)

        v = self.value(x)
        output = p @ v
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, block_size, head_size, n_head, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class Expert(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
