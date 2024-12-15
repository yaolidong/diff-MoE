import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        # 处理2维输入，添加序列维度
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
        if len(context.shape) == 2:
            context = context.unsqueeze(1)  # [B, D] -> [B, 1, D]
    
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 如果输入是2维的，则输出也应该是2维的
        if len(x.shape) == 2:
            out = out.squeeze(1)  # [B, 1, D] -> [B, D] 
        return self.to_out(out)