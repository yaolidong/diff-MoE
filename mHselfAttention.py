import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, n_embd, n_head, head_size, dropout=0.1, 
                 attention_type='normal', use_rotary=False, use_flash=True):
        super().__init__()
        self.n_head = n_head
        self.head_size = head_size
        self.attention_type = attention_type
        self.use_rotary = use_rotary
        self.use_flash = use_flash and torch.cuda.is_available()
        
        # QKV投影
        self.q_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.k_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        self.v_proj = nn.Linear(n_embd, n_head * head_size, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(n_head * head_size, n_embd)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # 相对位置编码
        if attention_type == 'relative':
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * seq_len - 1, head_size) / math.sqrt(head_size))
        
        # 旋转位置编码
        if use_rotary:
            self.rotary_emb = RotaryEmbedding(head_size)
            
        # 注意力缩放因子
        self.scale = head_size ** -0.5
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
            
    def _split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.n_head, self.head_size)
        return x.transpose(1, 2)  # B, nh, T, hs
        
    def _merge_heads(self, x):
        B, nh, T, hs = x.shape
        x = x.transpose(1, 2)  # B, T, nh, hs
        return x.reshape(B, T, nh * hs)
    
    def _rel_pos_bias(self, seq_len):
        # 生成相对位置索引
        pos_idx = torch.arange(seq_len, device=self.rel_pos_emb.device)
        rel_pos_idx = pos_idx.unsqueeze(0) - pos_idx.unsqueeze(1)
        rel_pos_idx = rel_pos_idx + (seq_len - 1)  # 偏移使所有索引为正
        
        # 获取相对位置编码并扩展到所有注意力头
        rel_pos = self.rel_pos_emb[rel_pos_idx]  # [seq_len, seq_len, head_size]
        rel_pos = rel_pos.unsqueeze(0).expand(self.n_head, -1, -1, -1)  # [n_head, seq_len, seq_len, head_size]
        return rel_pos
        
    def forward(self, x, mask: Optional[torch.Tensor] = None, 
                past_key_value: Optional[tuple] = None):
        B, T, C = x.shape
        
        # 计算QKV
        q = self.q_proj(x)  # B, T, nh*hs
        k = self.k_proj(x)  # B, T, nh*hs
        v = self.v_proj(x)  # B, T, nh*hs
        
        # 分离头
        q = self._split_heads(q)  # B, nh, T, hs
        k = self._split_heads(k)  # B, nh, T, hs
        v = self._split_heads(v)  # B, nh, T, hs
        
        # 应用旋转位置编码
        if self.use_rotary:
            q, k = self.rotary_emb(q, k)
            
        # 处理past key value
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B, nh, T, T
        
        # 添加相对位置偏置
        if self.attention_type == 'relative':
            rel_pos_bias = self._rel_pos_bias(T)  # [n_head, T, T, head_size]
            # 计算注意力分数
            rel_attn = torch.einsum('bhid,hijd->bhij', q, rel_pos_bias)
            attn_scores = attn_scores + rel_attn
            
        # 应用mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 计算输出
        if self.use_flash and not self.training:
            # 使用Flash Attention (仅在推理时)
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0 if not self.training else self.attn_dropout.p
            )
        else:
            # 标准注意力计算
            attn_output = torch.matmul(attn_weights, v)
        
        # 合并头
        out = self._merge_heads(attn_output)
        
        # 输出投影
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        # 返回输出和注意力权重
        return out, attn_weights

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        # 应用旋转
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
        
    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

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