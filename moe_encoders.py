import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        
        # 确保 input_dim 能被 num_heads 整除
        self.adjusted_dim = ((input_dim - 1) // num_heads + 1) * num_heads
        self.input_projection = nn.Linear(input_dim, self.adjusted_dim)
        
        # 使用调整后的维度
        self.multi_head_attention = nn.MultiheadAttention(self.adjusted_dim, num_heads)
        
        self.gate = nn.Linear(self.adjusted_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adjusted_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # 投影到调整后的维度
        x = self.input_projection(x.float())  # 确保输入是浮点类型
        
        # 调试代码：打印 x 的形状
        # print(f"Shape after input_projection: {x.shape}")
        
        # 调整输入形状以适应 MultiheadAttention
        x = x.unsqueeze(0)  # 添加序列长度维度
        
        # 通过多头注意力机制
        x, _ = self.multi_head_attention(x, x, x)
        
        # 去除序列长度维度
        x = x.squeeze(0)
        
        # 调试代码：打印 x 的形状
        # print(f"Shape after multi_head_attention: {x.shape}")
        
        # 门控层
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_outputs = torch.zeros(x.size(0), self.output_dim, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_inputs = x[mask.any(dim=1)]
                expert_output = expert(expert_inputs)
                expert_outputs[mask.any(dim=1)] += expert_output * top_k_probs[mask][:, None]
        
        return expert_outputs

class ImageMoE(nn.Module):
    def __init__(self, input_dim=784, output_dim=128, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.first_moe = MoELayer(input_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.cls = nn.Linear(output_dim, output_dim)
        self.vector = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        first_output = self.first_moe(x)
        first_vector = self.vector(first_output)
        
        second_output = self.second_moe(first_vector)
        second_vector = self.vector(second_output)
        cls_first = self.cls(first_output)
        cls_second = self.cls(second_output)
        
        return first_vector, second_vector, cls_first, cls_second

class TextMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, output_dim=128, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.first_moe = MoELayer(embed_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.cls = nn.Linear(output_dim, output_dim)
        self.vector = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = x.float()  
        first_output = self.first_moe(x)
        first_vector = self.vector(first_output)
        
        second_output = self.second_moe(first_vector)
        second_vector = self.vector(second_output)
        cls_first = self.cls(first_output)
        cls_second = self.cls(second_output)
        
        return first_vector, second_vector, cls_first, cls_second

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, y):
        batch_size = x.size(0)

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(y).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.out(out)