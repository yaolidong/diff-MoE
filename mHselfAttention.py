import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, seq_len, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        # 计算注意力分数
        att = (query @ key.transpose(-2, -1)) * C**-0.5
        
        # 应用因果掩码
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 应用softmax
        att = F.softmax(att, dim=-1)
        
        # 应用dropout
        att = self.dropout(att)

        # 计算输出
        out = att @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_len, n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)  
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        output = self.proj(output)
        output = self.dropout(output)
        return output


class Expert(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
