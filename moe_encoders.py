import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encode import positional_encoding
import mHselfAttention
from NoisyTopkRouter import NoisyTopkRouter

class SparseMoE(nn.Module):
    def __init__(self, n_embd, top_k, num_experts, dropout):
        super(SparseMoE, self).__init__()
        self.experts = nn.ModuleList([mHselfAttention.Expert(n_embd, dropout) for _ in range(num_experts)])
        self.router = NoisyTopkRouter(n_embd, top_k, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)  # 路由器输出top_k 专家softmax概率和专家索引
        final_output = torch.zeros_like(x)  # 初始化最终输出

        # 三维转二维
        flat_x = x.view(-1, x.size(-1))  # flatten x, x.size(-1)最后一个维度数，-1表示自动计算
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))  # gating_output.size(-1) = num_experts

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)  # 创建一个掩码，指示哪些位置由专家i处理
            flat_mask = expert_mask.view(-1)  # 沿最后的维度展平

            if flat_mask.any():
                expert_input = flat_x[flat_mask]  # 从输入中提取专家i需要处理的数据
                expert_output = expert(expert_input)  # 专家i处理tokens并输出

                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)  # 从路由器输出矩阵中提取专家i的概率
                weighted_output = expert_output * gating_scores  # 乘以概率

                final_output[expert_mask] += weighted_output.squeeze(1)  # 将加权输出添加到最终输出中

        return final_output

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.output_dim = output_dim
        
        self.adjusted_dim = ((input_dim - 1) // num_heads + 1) * num_heads
        self.input_projection = nn.Linear(input_dim, self.adjusted_dim)
        
        # 修改这里以使用mHselfAttention中的MultiHeadAttention
        self.multi_head_attention = mHselfAttention.MultiHeadAttention(
            block_size=self.adjusted_dim,
            head_size=self.adjusted_dim // num_heads,
            n_head=num_heads,
            n_embd=self.adjusted_dim,
            dropout=0.1
        )
        
        self.sparse_moe = SparseMoE(self.adjusted_dim, top_k, num_experts, dropout=0.1)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, attention_mask=None):
        x = self.input_projection(x.float())
        x = self.dropout(x)

        # 注意：mHselfAttention.MultiHeadAttention 不接受 attention_mask 参数
        x = self.multi_head_attention(x)
        
        x = self.dropout(x)
        
        expert_outputs = self.sparse_moe(x)
        
        expert_outputs = self.layer_norm(expert_outputs)
        expert_outputs = self.dropout(expert_outputs)
    
        return expert_outputs, None  # 返回 None 作为 attn_weights，因为 mHselfAttention.MultiHeadAttention 不返回注意力权重

class ImageMoE(nn.Module):
    def __init__(self, img_size=28, patch_size=4, input_dim=784, output_dim=128, num_experts=10, top_k=2, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.output_dim = output_dim
        self.patch_embeddings = nn.Linear(self.patch_dim, output_dim)
        self.first_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.vector = nn.Linear(output_dim, output_dim)
        self.global_vector = nn.Linear(output_dim * self.num_patches, output_dim)
        self.cls = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device  # 获取输入张量的设备

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, c, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(b, -1, self.patch_dim)
        
        # 应用patch嵌入
        x = self.patch_embeddings(x)
        
        # 添加位置编码
        pe = positional_encoding(b, self.num_patches, self.output_dim).to(device)  # 确保位置编码在正确的设备上
        x = x + pe
    
        first_output, first_attn_weights = self.first_moe(x)
        first_vector = self.vector(first_output)
        
        second_output, second_attn_weights = self.second_moe(first_vector)
        second_vector = self.vector(second_output)
        
        # 确保second_attn_weights的维度与second_vector匹配
        second_attn_weights = second_attn_weights.view(b, -1, 1)
        
        # 使用注意力权重来计算全局向量
        global_vector = torch.sum(second_vector * second_attn_weights.mean(dim=1, keepdim=True), dim=1)
        
        # 生成CLS向量
        cls_vector = self.cls(global_vector)

        return first_vector, second_vector, global_vector, cls_vector

class TextMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, output_dim=128, num_experts=10, top_k=2, num_heads=8, max_seq_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.first_moe = MoELayer(embed_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.vector = nn.Linear(output_dim, output_dim)
        self.sentence_vector = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):  
        x = self.embedding(input_ids)
        b, seq_length, embed_dim = x.shape
        
        # 添加位置编码
        pe = positional_encoding(b, seq_length, embed_dim).to(x.device)
        x = x + pe
        x = self.dropout(x)
        
        print(f"Embedding stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        first_output, first_attn_weights = self.first_moe(x, attention_mask)
        first_vector = self.vector(first_output)
        first_vector = self.dropout(first_vector)
        
        print(f"First MoE output stats: mean={first_vector.mean().item():.4f}, std={first_vector.std().item():.4f}")
        print(f"First attn weights stats: mean={first_attn_weights.mean().item():.4f}, std={first_attn_weights.std().item():.4f}")
        
        second_output, second_attn_weights = self.second_moe(first_vector, attention_mask)
        second_vector = self.vector(second_output)
        second_vector = self.dropout(second_vector)
        
        print(f"Second MoE output stats: mean={second_vector.mean().item():.4f}, std={second_vector.std().item():.4f}")
        print(f"Second attn weights stats: mean={second_attn_weights.mean().item():.4f}, std={second_attn_weights.std().item():.4f}")
        
        # 使用注意力权重和attention_mask来计算句子向量
        mask = attention_mask.unsqueeze(-1).float()
        weighted_second_vector = second_vector * second_attn_weights.unsqueeze(-1) * mask
        sentence_vector = torch.sum(weighted_second_vector, dim=1) / torch.sum(mask, dim=1)
        
        sentence_vector = self.sentence_vector(sentence_vector)
        sentence_vector = self.layer_norm(sentence_vector)
        sentence_vector = self.dropout(sentence_vector)
        
        print(f"Final sentence vector stats: mean={sentence_vector.mean().item():.4f}, std={sentence_vector.std().item():.4f}")
        
        if b > 1:
            print(f"Are all sentence vectors the same? {torch.allclose(sentence_vector[0], sentence_vector[1], atol=1e-6)}")
            print(f"Sentence vector difference: {(sentence_vector[0] - sentence_vector[1]).abs().mean().item():.6f}")
        
        return first_vector, second_vector, sentence_vector.mean(dim=1)
