import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        
        self.adjusted_dim = ((input_dim - 1) // num_heads + 1) * num_heads
        self.input_projection = nn.Linear(input_dim, self.adjusted_dim)
        
        self.multi_head_attention = nn.MultiheadAttention(self.adjusted_dim, num_heads)
        
        self.gate = nn.Linear(self.adjusted_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.adjusted_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x, attention_mask=None):
        x = self.input_projection(x.float())
        print(f"x的维度: {x.shape}")
        if attention_mask is not None:
            # 文本输入的情况
            x = x.transpose(0, 1)
            attention_mask = attention_mask.to(dtype=torch.bool)
            x, attn_weights = self.multi_head_attention(x, x, x, key_padding_mask=attention_mask)
            x = x.transpose(0, 1)
        else:
            # 图像输入的情况
            x, attn_weights = self.multi_head_attention(x, x, x)
        
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        expert_outputs = torch.zeros(*x.shape[:-1], self.output_dim, device=x.device)
        for i, expert in enumerate(self.experts):
            mask = top_k_indices == i
            if mask.any():
                expert_inputs = x[mask.any(dim=-1)]
                expert_output = expert(expert_inputs)
                expert_outputs[mask.any(dim=-1)] += expert_output * top_k_probs[mask].unsqueeze(-1)
        
        return expert_outputs, attn_weights

class ImageMoE(nn.Module):
    def __init__(self, img_size=28, patch_size=4, input_dim=784, output_dim=128, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, output_dim))
        self.patch_embeddings = nn.Linear(self.patch_dim, output_dim)
        
        self.first_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.cls = nn.Linear(output_dim, output_dim)
        self.vector = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        # 将图像分成patch
        b, c, h, w = x.shape
        x = x.view(b, c, self.img_size, self.img_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, c, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.patch_dim)
        
        # 添加位置编码
        x = self.patch_embeddings(x) + self.position_embeddings

        print(f"图像维度: {x.shape}")
        first_output, _ = self.first_moe(x)
        first_vector = self.vector(first_output)
        print(f"first_output维度: {first_output.shape}")
        second_output, _ = self.second_moe(first_vector)
        print(f"second_output维度: {second_output.shape}")
        second_vector = self.vector(second_output)
        cls_first = self.cls(first_output)
        cls_second = self.cls(second_output)

        
        return first_vector, second_vector, cls_first, cls_second
class TextMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, output_dim=128, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.first_moe = MoELayer(embed_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.cls = nn.Linear(output_dim, output_dim)
        self.vector = nn.Linear(output_dim, output_dim)
    
    def forward(self, input_ids, attention_mask):  
        x = self.embedding(input_ids)
        
        first_output, _ = self.first_moe(x, attention_mask)
        first_vector = self.vector(first_output)
        
        second_output, _ = self.second_moe(first_vector, attention_mask)
        second_vector = self.vector(second_output)
        cls_first = self.cls(first_output)
        cls_second = self.cls(second_output)
        
        
        return first_vector, second_vector, cls_first, cls_second

