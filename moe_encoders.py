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
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, attention_mask=None):
        x = self.input_projection(x.float())
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
        
        expert_outputs = self.layer_norm(expert_outputs)
        
        # 确保attn_weights的维度与expert_outputs匹配
        attn_weights = attn_weights.view(*expert_outputs.shape[:-1], -1)
        attn_weights = attn_weights.mean(dim=-1, keepdim=True)
        
        # 使用注意力权重对expert_outputs进行加权
        weighted_outputs = expert_outputs * attn_weights
        
        return weighted_outputs, attn_weights

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
        self.vector = nn.Linear(output_dim, output_dim)
        self.global_vector = nn.Linear(output_dim * self.num_patches, output_dim)
        self.cls = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, self.img_size, self.img_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, c, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.patch_dim)
        x = self.patch_embeddings(x)
        
        # 添加位置编码
        x = x + self.position_embeddings

        first_output, first_attn_weights = self.first_moe(x)
        first_vector = self.vector(first_output)
        
        second_output, second_attn_weights = self.second_moe(first_vector)
        second_vector = self.vector(second_output)
        
        # 确保second_attn_weights的维度与second_vector匹配
        second_attn_weights = second_attn_weights.view(b, -1, 1)
        
        # 使用注意力权重来计算全局向量
        global_vector = torch.sum(second_vector * second_attn_weights, dim=1)
        
        # 生成CLS向量
        cls_vector = self.cls(global_vector)

        return first_vector, second_vector, global_vector, cls_vector
class TextMoE(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, output_dim=128, num_experts=10, top_k=2, num_heads=8, max_seq_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.first_moe = MoELayer(embed_dim, output_dim, num_experts, top_k, num_heads)
        self.second_moe = MoELayer(output_dim, output_dim, num_experts, top_k, num_heads)
        self.vector = nn.Linear(output_dim, output_dim)
        self.sentence_vector = nn.Linear(output_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, input_ids, attention_mask):  
        x = self.embedding(input_ids)
        print(f"{x[0,:,:] == x[1,:,:]}")
        print(f"{x[1,:,:] == x[2,:,:]}")
        # 添加位置编码
        seq_length = x.size(1)
        x = x + self.position_embeddings[:, :seq_length, :]
        
        # 应用第一个MoE层
        first_output, first_attn_weights = self.first_moe(x, attention_mask)
        first_vector = self.vector(first_output)

        
        # 应用第二个MoE层
        second_output, second_attn_weights = self.second_moe(first_vector, attention_mask)
        second_vector = self.vector(second_output)
        
        # 使用注意力权重和attention_mask来计算句子向量
        mask = attention_mask.unsqueeze(-1).float()
        weighted_second_vector = second_vector * second_attn_weights.unsqueeze(-1) * mask
        sentence_vector = torch.sum(weighted_second_vector, dim=1) / torch.sum(mask, dim=1)
        
        sentence_vector = self.sentence_vector(sentence_vector)
        sentence_vector = self.layer_norm(sentence_vector)
        
        return first_vector, second_vector, sentence_vector
