import torch
import torch.nn as nn
from positional_encoder import get_positional_encoding
from mHselfAttention import MultiHeadAttention, Expert
from AttentiveRouter import AttentiveRouter
import matplotlib.pyplot as plt
import math
from einops import rearrange
import torch.nn.functional as F

class SparseMoE(nn.Module):
    def __init__(self, n_embd=512, expert_size=None, top_k=2, num_experts=10, capacity_factor=1.2):
        super(SparseMoE, self).__init__()
        expert_size = expert_size or (n_embd * 4)  # 如果未指定expert_size，则默认为n_embd的4倍
        self.experts = nn.ModuleList([Expert(n_embd, expansion_factor=expert_size//n_embd) for _ in range(num_experts)])
        self.router = AttentiveRouter(n_embd, top_k, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts
        
        # 专家使用率监控
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.zeros(1))

    def _update_expert_stats(self, indices):
        for i in range(self.num_experts):
            self.expert_counts[i] += (indices == i).sum().item()
        self.total_tokens += indices.numel()

    def get_expert_usage_stats(self):
        if self.total_tokens.item() == 0:
            return torch.zeros_like(self.expert_counts)
        return self.expert_counts / self.total_tokens.item()

    def reset_expert_stats(self):
        self.expert_counts.zero_()
        self.total_tokens.zero_()

    def forward(self, x):
        # 获取路由结果
        expert_mask, indices, router_loss = self.router(x)
        
        # 更新专家使用统计
        self._update_expert_stats(indices)
        
        # 获取输入的维度
        B, S, D = x.shape
        
        # 创建输出张量
        expert_outputs = []
        expert_activations = []  # 用于可视化的激活图
        
        # 并行处理所有专家
        for i, expert in enumerate(self.experts):
            # 扩展expert_mask以匹配输入维度
            mask = expert_mask[..., i].unsqueeze(-1)
            # 计算专家输出并应用mask
            expert_output = expert(x)
            masked_output = expert_output * mask
            expert_outputs.append(masked_output)
            
            # 计算专家激活图
            activation = expert_output.mean(dim=-1)  # 对特征维度取平均
            expert_activations.append(activation)
            
        # 堆叠所有专家的输出
        combined_expert_outputs = torch.stack(expert_outputs)
        expert_activations = torch.stack(expert_activations)  # [num_experts, batch, seq_len]
        
        # 合并所有专家的输出
        final_output = combined_expert_outputs.sum(dim=0)
        
        # 获取路由器的注意力权重
        router_attn = self.router.get_attention_weights()  # 获取路由器的注意力权重
        
        # 返回所有需要的信息
        return {
            'output': final_output,
            'expert_outputs': expert_outputs,  # 每个专家的原始输出
            'expert_activations': expert_activations,  # 用于可视化的激活图
            'gating_output': expert_mask,  # 门控输出（专家选择概率）
            'router_loss': router_loss,  # 路由损失
            'router_attention': router_attn,  # 路由器的注意力权重
            'indices': indices  # 选择的专家索引
        }

class ImageMoE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256, 
                 num_shared_experts=6, num_modality_specific_experts=1,
                 top_k=2, num_heads=8, dropout=0.1, expert_capacity_factor=1.25):
        super().__init__()
        
        # 图像和patch参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch嵌入层
        self.patch_embeddings = nn.Linear(self.patch_dim, embed_dim)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) / math.sqrt(embed_dim))
        
        # 自注意力层
        self.sa = MultiHeadAttention(
            seq_len=self.num_patches,
            n_embd=embed_dim,
            n_head=num_heads,
            head_size=embed_dim // num_heads,
            dropout=dropout
        )
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
        # 共享专家层
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, embed_dim * 4)
            for _ in range(num_shared_experts)
        ])
        
        # 图像特异性专家
        self.image_specific_expert = Expert(embed_dim, embed_dim * 4)
        
        # 专家路由器
        self.router = AttentiveRouter(
            embed_dim=embed_dim,
            num_experts=num_shared_experts + 1,  # 共享专家 + 1个图像特异性专家
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        # 分类头
        self.classification = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 统计信息
        self.expert_usage_stats = {}
        
    def forward(self, x):
        # 输入处理和patch嵌入
        b, c, h, w = x.shape
        
        # 确保输入图像大小正确
        if h != self.img_size or w != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        # 处理通道数
        if c != self.in_channels:
            if c == 1 and self.in_channels == 3:
                x = x.repeat(1, 3, 1, 1)
            elif c == 3 and self.in_channels == 1:
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Patch操作
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                          p1=self.patch_size, p2=self.patch_size)
        
        # Patch嵌入
        x = self.patch_embeddings(patches)
        
        # 添加位置编码和注意力
        x = x + self.pos_embedding
        x_ln = self.ln1(x)
        sa_out, attn_weights = self.sa(x_ln)
        x = x + sa_out
        x = self.dropout(x)
        
        # 路由到专家
        router_outputs = self.router(self.ln2(x))
        expert_outputs = torch.zeros_like(x)
        
        # 处理共享专家
        for i, expert in enumerate(self.shared_experts):
            mask = router_outputs['expert_masks'][:, :, i:i+1]
            expert_output = expert(x)
            expert_outputs += mask * expert_output
        
        # 处理图像特异性专家
        img_mask = router_outputs['expert_masks'][:, :, -1:]
        img_specific_output = self.image_specific_expert(x)
        expert_outputs += img_mask * img_specific_output
        
        # 最终处理
        expert_outputs = self.ln3(expert_outputs)
        feature_vector = expert_outputs.mean(dim=1)
        cls = self.classification(feature_vector)
        
        # 更新专家使用统计
        self._update_expert_stats(router_outputs)
        
        return {
            'feature_vector': feature_vector,
            'logits': cls,
            'router_loss': router_outputs['router_loss'],
            'attention_weights': attn_weights,
            'expert_outputs': expert_outputs,
            'expert_masks': router_outputs['expert_masks'],
            'patches': patches,
            'patch_embeddings': x
        }
    
    def _update_expert_stats(self, router_outputs):
        """更新专家使用统计"""
        expert_masks = router_outputs['expert_masks']
        expert_usage = expert_masks.mean(dim=[0, 1])
        self.expert_usage_stats = {
            'shared_experts': expert_usage[:-1],
            'image_specific': expert_usage[-1]
        }
    
    def get_expert_stats(self):
        """获取专家使用统计"""
        return self.expert_usage_stats
    
    def reset_expert_stats(self):
        """重置专家使用统计"""
        self.expert_usage_stats = {}

class TextMoE(nn.Module):
    def __init__(self, vocab_size, seq_length=16, embed_dim=1024,
                 num_shared_experts=6, num_modality_specific_experts=1,
                 top_k=2, expert_capacity_factor=1.25):
        super().__init__()
        self.head_size = embed_dim//8
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 使用可学习的位置编码
        self.pos_embedding = get_positional_encoding('learnable', embed_dim, seq_length)
        
        # 使用改进的多注意力
        self.sa = MultiHeadAttention(
            seq_len=seq_length,
            n_embd=embed_dim,
            n_head=8,
            head_size=self.head_size,
            dropout=0.1,
            attention_type='relative',
            use_rotary=True
        )
        
        # Layer Norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        
        # 共享专家层
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, embed_dim * 4)
            for _ in range(num_shared_experts)
        ])
        
        # 文本特异性专家
        self.text_specific_expert = Expert(embed_dim, embed_dim * 4)
        
        # 专家路由器
        self.router = AttentiveRouter(
            embed_dim=embed_dim,
            num_experts=num_shared_experts + 1,  # 共享专家 + 1个文本特异性专家
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(embed_dim, embed_dim)
        
        # 统计信息
        self.expert_usage_stats = {}
    
    def forward(self, input_ids, attention_mask=None):
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 添加位置编码和注意力
        x = self.pos_embedding(x)
        sa_out, attn_weights = self.sa(self.ln1(x))
        x = x + sa_out
        x = self.dropout(x)
        
        # 路由到专家
        router_outputs = self.router(self.ln2(x))
        expert_outputs = torch.zeros_like(x)
        
        # 处理共享专家
        for i, expert in enumerate(self.shared_experts):
            mask = router_outputs['expert_masks'][:, :, i:i+1]
            expert_output = expert(x)
            expert_outputs += mask * expert_output
        
        # 处理文本特异性专家
        text_mask = router_outputs['expert_masks'][:, :, -1:]
        text_specific_output = self.text_specific_expert(x)
        expert_outputs += text_mask * text_specific_output
        
        # 最终处理
        expert_outputs = self.ln3(expert_outputs)
        feature_vector = expert_outputs.mean(dim=1)
        cls = self.classification(feature_vector)
        
        # 更新专家使用统计
        self._update_expert_stats(router_outputs)
        
        return {
            'feature_vector': feature_vector,
            'logits': cls,
            'router_loss': router_outputs['router_loss'],
            'attention_weights': attn_weights,
            'expert_outputs': expert_outputs,
            'expert_masks': router_outputs['expert_masks']
        }
    
    def _update_expert_stats(self, router_outputs):
        """更新专家使用统计"""
        expert_masks = router_outputs['expert_masks']
        expert_usage = expert_masks.mean(dim=[0, 1])
        self.expert_usage_stats = {
            'shared_experts': expert_usage[:-1],
            'text_specific': expert_usage[-1]
        }
    
    def get_expert_stats(self):
        """获取专家使用统计"""
        return self.expert_usage_stats
    
    def reset_expert_stats(self):
        """重置专家使用统计"""
        self.expert_usage_stats = {}

class UnifiedModalEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 vocab_size=30522, seq_length=77, embed_dim=1024,
                 num_shared_experts=6, num_modality_specific_experts=1,
                 top_k=2, num_layers=6, expert_capacity_factor=1.25):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = ImageMoE(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_shared_experts=num_shared_experts,
            num_modality_specific_experts=num_modality_specific_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        # 文本编码器
        self.text_encoder = TextMoE(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embed_dim=embed_dim,
            num_shared_experts=num_shared_experts,
            num_modality_specific_experts=num_modality_specific_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        # 模态类型嵌入
        self.modality_embeddings = nn.Parameter(torch.randn(2, embed_dim))
        
        # 共享专家层
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, embed_dim * 4)
            for _ in range(num_shared_experts)
        ])
        
        # 模态特异性专家
        self.image_specific_expert = Expert(embed_dim, embed_dim * 4)
        self.text_specific_expert = Expert(embed_dim, embed_dim * 4)
        
        # 专家路由器
        self.router = AttentiveRouter(
            embed_dim=embed_dim,
            num_experts=num_shared_experts + 2,  # 共享专家 + 2个模态特异性专家
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor
        )
        
        # 统计信息
        self.expert_usage_stats = {
            'image_encoder': {},
            'text_encoder': {},
            'cross_modal_layers': {}
        }
        
    def forward(self, image=None, input_ids=None, attention_mask=None):
        batch_features = []
        total_router_loss = 0
        
        if image is not None:
            # 图像特征提取
            img_outputs = self.image_encoder(image)
            img_features = img_outputs['feature_vector']
            total_router_loss += img_outputs['router_loss']
            
            # 添加模态类型嵌入
            img_features = img_features + self.modality_embeddings[0]
            batch_features.append(img_features)
        
        if input_ids is not None:
            # 文本特征提取
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_features = text_outputs['feature_vector']
            total_router_loss += text_outputs['router_loss']
            
            # 添加模态类型嵌入
            text_features = text_features + self.modality_embeddings[1]
            batch_features.append(text_features)
        
        if not batch_features:
            raise ValueError("至少需要提供一种模态的输入")
        
        # 合并特征
        combined_features = torch.cat(batch_features, dim=1)
        
        # 路由到专家
        router_outputs = self.router(combined_features)
        expert_outputs = torch.zeros_like(combined_features)
        
        # 处理共享专家
        for i, expert in enumerate(self.shared_experts):
            mask = router_outputs['expert_masks'][:, :, i:i+1]
            expert_output = expert(combined_features)
            expert_outputs += mask * expert_output
        
        # 处理模态特异性专家
        # 图像特异性专家
        if image is not None:
            img_mask = router_outputs['expert_masks'][:, :, -2:-1]
            img_specific_output = self.image_specific_expert(combined_features)
            expert_outputs += img_mask * img_specific_output
        
        # 文本特异性专家
        if input_ids is not None:
            text_mask = router_outputs['expert_masks'][:, :, -1:]
            text_specific_output = self.text_specific_expert(combined_features)
            expert_outputs += text_mask * text_specific_output
        
        # 更新专家使用统计
        self._update_expert_stats(router_outputs)
        
        return {
            'feature_vector': expert_outputs,
            'router_loss': total_router_loss + router_outputs['router_loss'],
            'expert_outputs': expert_outputs,
            'attention_weights': router_outputs['attention_weights'],
            'expert_masks': router_outputs['expert_masks']
        }
    
    def _update_expert_stats(self, router_outputs):
        """更新专家使用统计"""
        expert_masks = router_outputs['expert_masks']
        expert_usage = expert_masks.mean(dim=[0, 1])  # [num_experts]
        
        # 分别统计共享专家和特异性专家的使用情况
        self.expert_usage_stats['cross_modal_layers']['shared_experts'] = expert_usage[:-2]
        self.expert_usage_stats['cross_modal_layers']['image_specific'] = expert_usage[-2]
        self.expert_usage_stats['cross_modal_layers']['text_specific'] = expert_usage[-1]
    
    def get_expert_stats(self):
        """获取专家使用统计"""
        return self.expert_usage_stats
    
    def reset_expert_stats(self):
        """重置专家使用统计"""
        for key in self.expert_usage_stats:
            self.expert_usage_stats[key] = {} 