import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建固定的位置编码
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_seq_length = max_seq_length
        self.pe = nn.Parameter(torch.randn(1, max_seq_length, d_model) * 0.1)
        
    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            raise ValueError(f"序列长度 {seq_len} 超过了最大长度 {self.max_seq_length}")
        pos_emb = self.pe[:, :seq_len]
        x = x + pos_emb
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512, dropout=0.1, max_relative_position=32):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_position = max_relative_position
        
        # 创建相对位置编码表
        self.relative_positions_embeddings = nn.Parameter(
            torch.randn(max_relative_position * 2 + 1, d_model) * 0.1
        )
        
    def _get_relative_positions(self, length):
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat
        
    def forward(self, x):
        seq_length = x.size(1)
        relative_positions = self._get_relative_positions(seq_length)
        relative_positions = relative_positions.to(x.device)
        
        # 获取相对位置编码
        relative_pos_embeddings = self.relative_positions_embeddings[relative_positions]
        
        return self.dropout(x), relative_pos_embeddings

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def _get_rotary_embeddings(self, seq_length, device):
        t = torch.arange(seq_length, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
        
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
        
    def forward(self, q, k):
        seq_length = q.shape[1]
        cos, sin = self._get_rotary_embeddings(seq_length, q.device)
        
        # 应用旋转
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

def get_positional_encoding(encoding_type, d_model, max_seq_length=512, dropout=0.1):
    if encoding_type == 'fixed':
        return PositionalEncoding(d_model, max_seq_length, dropout)
    elif encoding_type == 'learnable':
        return LearnablePositionalEncoding(d_model, max_seq_length, dropout)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(d_model, max_seq_length, dropout)
    elif encoding_type == 'rotary':
        return RotaryPositionalEncoding(d_model, max_seq_length)
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}") 