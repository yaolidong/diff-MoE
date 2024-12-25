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
            
        # 堆叠��有专家的输出
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
                 num_experts=8, top_k=2, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 图像和patch参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2  # 计算patch数量
        self.patch_dim = in_channels * patch_size * patch_size  # 每个patch的维度
        
        print(f"初始化参数:")
        print(f"图像大小: {img_size}x{img_size}")
        print(f"Patch大小: {patch_size}x{patch_size}")
        print(f"输入通道数: {in_channels}")
        print(f"Patch数量: {self.num_patches}")
        print(f"每个Patch维度: {self.patch_dim}")
        
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
        
        # MoE层
        self.first_moe = SparseMoE(
            n_embd=embed_dim,
            expert_size=embed_dim * 4,
            num_experts=num_experts,
            top_k=top_k
        )
        
        self.second_moe = SparseMoE(
            n_embd=embed_dim,
            expert_size=embed_dim * 4,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 分类头
        self.classification = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def get_expert_stats(self):
        """获取专家使用统计"""
        return {
            'first_layer': self.first_moe.get_expert_usage_stats(),
            'second_layer': self.second_moe.get_expert_usage_stats()
        }
    
    def reset_expert_stats(self):
        """重置专家使用统计"""
        self.first_moe.reset_expert_stats()
        self.second_moe.reset_expert_stats()

    def forward(self, x):
        b, c, h, w = x.shape
        print(f"Input shape: {x.shape}")  # 调试信息
        
        # 确保输入图像大小正确
        if h != self.img_size or w != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            print(f"After resize: {x.shape}")  # 调试信息
        
        # 如果输入通道数不匹配，调整通道数
        if c != self.in_channels:
            if c == 1 and self.in_channels == 3:
                # 如果输入是��通道但模型期望三通道，复制到三个通道
                x = x.repeat(1, 3, 1, 1)
            elif c == 3 and self.in_channels == 1:
                # 如果输入是三通道但模型期望单通道，转换为灰度图
                x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            print(f"After channel adjustment: {x.shape}")  # 调试信息
        
        # 使用einops进行patch操作
        patches = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                          p1=self.patch_size, p2=self.patch_size)
        print(f"After patch reshape: {patches.shape}")  # 调试信息
        
        # 验证patch数量和维度
        assert patches.shape[1] == self.num_patches, f'Patch数量不匹配: 得到 {patches.shape[1]}, 期望 {self.num_patches}'
        assert patches.shape[2] == self.patch_dim, f'Patch维度不匹配: 得到 {patches.shape[2]}, 期望 {self.patch_dim}'
        
        # Patch嵌入
        x = self.patch_embeddings(patches)  # [B, num_patches, embed_dim]
        print(f"After embedding: {x.shape}")  # 调试信息
        
        # 添加位置编码和注意力
        x = x + self.pos_embedding
        x_ln = self.ln1(x)
        sa_out, attn_weights = self.sa(x_ln)
        x = x + sa_out
        x = self.dropout(x)
        
        # MoE层
        first_moe_outputs = self.first_moe(self.ln2(x))
        second_moe_outputs = self.second_moe(self.ln3(first_moe_outputs['output']))
        
        # 生成特征向量和分类
        feature_vector = second_moe_outputs['output'].mean(dim=1)
        cls = self.classification(feature_vector)
        
        # 合并路由损失
        total_router_loss = first_moe_outputs['router_loss'] + second_moe_outputs['router_loss']

        # 处理注意力权重
        if attn_weights is not None:
            # 如果是多头注意力，取平均
            if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
                attn_weights = attn_weights.mean(dim=1)  # 对所有头取平均
            # 确保注意力权重是概率分布
            attn_weights = F.softmax(attn_weights, dim=-1)

        # 处理专家激活图
        expert_activations = []
        for expert_output in first_moe_outputs['expert_outputs']:
            # 计算每个专家的激活图
            activation = expert_output.mean(dim=-1)  # 对特征维度取平均
            expert_activations.append(activation)
        expert_activations = torch.stack(expert_activations)  # [num_experts, batch, seq_len]

        return {
            'feature_vector': feature_vector,
            'logits': cls,
            'router_loss': total_router_loss,
            'attention_weights': attn_weights,  # 注意力权重
            'expert_outputs': expert_activations,  # 专家激活图
            'first_expert_outputs': first_moe_outputs['expert_outputs'],  # 第一层专家原始输出
            'second_expert_outputs': second_moe_outputs['expert_outputs'],  # 第二层专家原始输出
            'first_gating_output': first_moe_outputs['gating_output'],  # 第一层门控输出
            'second_gating_output': second_moe_outputs['gating_output'],  # 第二层门控输出
            'first_router_attention': first_moe_outputs['router_attention'],  # 第一层路由器注意力
            'second_router_attention': second_moe_outputs['router_attention'],  # 第二层路由器注意力
            'patches': patches,  # 原始patch
            'patch_embeddings': x  # patch嵌入
        }

class TextMoE(nn.Module):
    def __init__(self, vocab_size, seq_length=16, embed_dim=1024, num_experts=10, top_k=2, capacity_factor=1.2):
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
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.first_moe = SparseMoE(
            n_embd=embed_dim,
            expert_size=embed_dim * 4,
            top_k=top_k,
            num_experts=num_experts,
            capacity_factor=capacity_factor
        )
        self.second_moe = SparseMoE(
            n_embd=embed_dim,
            expert_size=embed_dim * 4,
            top_k=top_k,
            num_experts=num_experts,
            capacity_factor=capacity_factor
        )
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(embed_dim, 10)
    
    def get_expert_stats(self):
        return {
            'first_layer': self.first_moe.get_expert_usage_stats(),
            'second_layer': self.second_moe.get_expert_usage_stats()
        }
    
    def reset_expert_stats(self):
        self.first_moe.reset_expert_stats()
        self.second_moe.reset_expert_stats()

    def forward(self, input_ids, attention_mask):  
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 添加位置编码和注意力
        x = self.pos_embedding(x)
        sa_out, attn_weights = self.sa(self.ln1(x))
        x = x + sa_out
        x = self.dropout(x)
        
        # MoE层
        first_moe_outputs = self.first_moe(self.ln2(x))
        second_moe_outputs = self.second_moe(self.ln3(first_moe_outputs['output']))
        
        # 生成特征向量和分类
        feature_vector = second_moe_outputs['output'].mean(dim=1)
        cls = self.classification(feature_vector)
        
        # 合并路由损失
        total_router_loss = first_moe_outputs['router_loss'] + second_moe_outputs['router_loss']
        
        return {
            'feature_vector': feature_vector,
            'logits': cls,
            'router_loss': total_router_loss,
            'attention_weights': attn_weights,  # 注意力权重
            'expert_outputs': first_moe_outputs['expert_activations'],  # 专家激活图
            'first_expert_outputs': first_moe_outputs['expert_outputs'],  # 第一层专家原始输出
            'second_expert_outputs': second_moe_outputs['expert_outputs'],  # 第二层专家原始输出
            'first_gating_output': first_moe_outputs['gating_output'],  # 第一层门控输出
            'second_gating_output': second_moe_outputs['gating_output'],  # 第二层门控输出
            'first_router_attention': first_moe_outputs['router_attention'],  # 第一层路由器注意力
            'second_router_attention': second_moe_outputs['router_attention']  # 第二层路由器注意力
        }

class UnifiedModalEncoder(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_channels=3,
                 vocab_size=30522,
                 seq_length=77,
                 embed_dim=1024,
                 num_experts=8,
                 top_k=2,
                 num_layers=6):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = ImageMoE(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 文本编码器
        self.text_encoder = TextMoE(
            vocab_size=vocab_size,
            seq_length=seq_length,
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 跨模态融合层
        self.cross_modal_layers = nn.ModuleList([
            SparseMoE(
                n_embd=embed_dim,
                expert_size=embed_dim * 4,
                top_k=top_k,
                num_experts=num_experts
            ) for _ in range(num_layers)
        ])
        
        # 模态类型嵌入
        self.modality_embeddings = nn.Parameter(torch.randn(2, 1, embed_dim))
        
        # Layer Norm和投影层
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def encode_image(self, image):
        # 获取图像特征
        outputs = self.image_encoder(image)
        img_features = outputs['feature_vector']
        img_router_loss = outputs['router_loss']
        
        # 添加模态类型嵌入
        img_features = img_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        img_features = img_features + self.modality_embeddings[0]
        return img_features, img_router_loss
        
    def encode_text(self, input_ids, attention_mask):
        # 获取文本特征
        outputs = self.text_encoder(input_ids, attention_mask)
        text_features = outputs['feature_vector']
        text_router_loss = outputs['router_loss']
        
        # 添加模态类型嵌入
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
        text_features = text_features + self.modality_embeddings[1]
        return text_features, text_router_loss
    
    def forward(self, image=None, input_ids=None, attention_mask=None):
        total_router_loss = 0
        features = []
        encoder_outputs = {}
        
        # 编码图像
        if image is not None:
            image_outputs = self.image_encoder(image)
            img_features = image_outputs['feature_vector'].unsqueeze(1)
            img_features = img_features + self.modality_embeddings[0]
            features.append(img_features)
            total_router_loss += image_outputs['router_loss']
            encoder_outputs.update({
                'attention_weights': image_outputs.get('attention_weights'),
                'expert_outputs': image_outputs.get('expert_outputs'),
                'first_expert_outputs': image_outputs.get('first_expert_outputs'),
                'second_expert_outputs': image_outputs.get('second_expert_outputs'),
                'first_gating_output': image_outputs.get('first_gating_output'),
                'second_gating_output': image_outputs.get('second_gating_output'),
                'first_router_attention': image_outputs.get('first_router_attention'),
                'second_router_attention': image_outputs.get('second_router_attention'),
                'patches': image_outputs.get('patches'),
                'patch_embeddings': image_outputs.get('patch_embeddings')
            })
            
        # 编码文本
        if input_ids is not None:
            text_outputs = self.text_encoder(input_ids, attention_mask)
            text_features = text_outputs['feature_vector'].unsqueeze(1)
            text_features = text_features + self.modality_embeddings[1]
            features.append(text_features)
            total_router_loss += text_outputs['router_loss']
            # 如果没有图像编码器的输出，使用文本编码器的输出
            if not encoder_outputs:
                encoder_outputs.update({
                    'attention_weights': text_outputs.get('attention_weights'),
                    'expert_outputs': text_outputs.get('expert_outputs'),
                    'first_expert_outputs': text_outputs.get('first_expert_outputs'),
                    'second_expert_outputs': text_outputs.get('second_expert_outputs'),
                    'first_gating_output': text_outputs.get('first_gating_output'),
                    'second_gating_output': text_outputs.get('second_gating_output'),
                    'first_router_attention': text_outputs.get('first_router_attention'),
                    'second_router_attention': text_outputs.get('second_router_attention')
                })
            
        if not features:
            raise ValueError("至少要一种模态的输入")
            
        # 合并特征
        x = torch.cat(features, dim=1)  # [batch_size, num_modalities, embed_dim]
        x = self.ln_pre(x)
        
        # 跨模态融合
        cross_modal_features = x
        cross_modal_outputs = []
        for layer in self.cross_modal_layers:
            layer_outputs = layer(cross_modal_features)
            cross_modal_features = layer_outputs['output']
            cross_modal_outputs.append(layer_outputs)
            total_router_loss += layer_outputs['router_loss']
            
        # 后处理
        unified_features = self.ln_post(cross_modal_features)
        logits = self.proj(unified_features)
        
        # 合并所有输出
        outputs = {
            'feature_vector': unified_features,
            'logits': logits.mean(dim=1),  # 对序列维度取平均
            'router_loss': total_router_loss,
            'cross_modal_outputs': cross_modal_outputs  # 包含每一层的详细输出
        }
        outputs.update(encoder_outputs)  # 添加编码器的输出
        
        return outputs
    
    def get_expert_stats(self):
        stats = {
            'image_encoder': self.image_encoder.get_expert_stats(),
            'text_encoder': self.text_encoder.get_expert_stats(),
            'cross_modal_layers': {}
        }
        for i, layer in enumerate(self.cross_modal_layers):
            stats['cross_modal_layers'][f'layer_{i}'] = layer.get_expert_usage_stats()
        return stats
    
    def reset_expert_stats(self):
        self.image_encoder.reset_expert_stats()
        self.text_encoder.reset_expert_stats()
        for layer in self.cross_modal_layers:
            layer.reset_expert_stats() 