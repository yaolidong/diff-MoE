import torch
import torch.nn as nn
from positional_encode import positional_encoding
import mHselfAttention
from NoisyTopkRouter import NoisyTopkRouter

class SparseMoE(nn.Module):
    def __init__(self, n_embd=512, top_k=2, num_experts=10):
        super(SparseMoE, self).__init__()
        self.experts = nn.ModuleList([mHselfAttention.Expert(n_embd) for _ in range(num_experts)])
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


class ImageMoE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=1024, num_experts=10, top_k=2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.output_dim = embed_dim
        self.head_size = embed_dim // 8

        self.patch_embeddings = nn.Linear(self.patch_dim, embed_dim)
        self.positional_encoding = positional_encoding(128, self.num_patches, embed_dim)
        self.sa = mHselfAttention.MultiHeadAttention(seq_len=self.num_patches, n_embd=embed_dim, n_head=8, head_size=self.head_size, dropout=0.1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.first_moe = SparseMoE(embed_dim, top_k, num_experts)
        self.second_moe = SparseMoE(embed_dim, top_k, num_experts)
        self.classification = nn.Linear(embed_dim, 10)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # 如果输入是单通道图像，将其扩展为三通道
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
            c = 3
        
        # 确保图像大小正确
        if h != self.img_size or w != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, c, -1, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(b, -1, self.patch_dim)
        
        # 应用patch嵌入
        x = self.patch_embeddings(x)
        x = x + self.sa(self.ln1(x))
        # 添加位置编码
        x = x + self.pos_embedding.to(x.device)
        x = self.dropout(x)
        first_output = self.first_moe(self.ln2(x))
        second_output = self.second_moe(self.ln3(x))
        # 生成特征向量
        feature_vector = second_output.mean(dim=1)  # 取平均值作为特征向量
        cls = self.classification(feature_vector)

        return first_output, second_output, feature_vector, cls

class TextMoE(nn.Module):
    def __init__(self, vocab_size, seq_length=16, embed_dim=1024, num_experts=10, top_k=2):
        super().__init__()
        self.head_size = embed_dim//8
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = positional_encoding(128, seq_length, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.sa = mHselfAttention.MultiHeadAttention(seq_len=seq_length, n_embd=embed_dim, n_head=8, head_size=self.head_size, dropout=0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.first_moe = SparseMoE(embed_dim, top_k, num_experts)
        self.second_moe = SparseMoE(embed_dim, top_k, num_experts)
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(embed_dim, 10)

    def forward(self, input_ids, attention_mask):  
        b = input_ids.shape[0]
        # 词嵌入[128,16,128]
        x = self.embedding(input_ids)
        x = x + self.sa(self.ln1(x))
        # 添加位置编码[128,16,128]
        x = x + self.pos_embedding.to(x.device)

        x = self.dropout(x)
        first_output = self.first_moe(self.ln2(x))
        second_output = self.second_moe(self.ln3(x))
        feature_vector = second_output.mean(dim=1)  # 取平均值作为特征向量
        cls = self.classification(feature_vector)
        return first_output, second_output, feature_vector, cls
