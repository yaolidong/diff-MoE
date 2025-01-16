import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, n_embd, expansion_factor=4, dropout=0.1, activation='gelu'):
        super().__init__()
        
        # 选择激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # FFN层
        self.fc1 = nn.Linear(n_embd, expansion_factor * n_embd)
        self.fc2 = nn.Linear(expansion_factor * n_embd, n_embd)
        
        # Layer Norm
        self.ln = nn.LayerNorm(n_embd)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        # 应用Layer Norm
        x = self.ln(x)
        
        # FFN第一层
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout1(h)
        
        # FFN第二层
        out = self.fc2(h)
        out = self.dropout2(out)
        
        return out