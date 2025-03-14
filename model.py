import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass
import os
import math
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.utils.checkpoint as checkpoint

# 添加Expert类
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

@dataclass
class ModelConfig:
    """模型配置类"""
    img_size: int = 28
    patch_size: int = 4
    in_channels: int = 1
    embed_dim: int = 512
    num_shared_experts: int = 4
    num_modality_specific_experts: int = 2
    top_k: int = 2
    num_heads: int = 8
    num_layers: int = 6
    num_classes: int = 10
    dropout: float = 0.1
    activation: str = 'gelu'
    use_bias: bool = True
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = False

class PatchEmbed(nn.Module):
    """图像分块嵌入层"""
    
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int
    ):
        """初始化
        
        Args:
            img_size: 输入图像大小
            patch_size: 分块大小
            in_channels: 输入图像通道数
            embed_dim: 嵌入维度
        """
        super().__init__()
        
        # 检查输入参数的有效性
        if not isinstance(img_size, (int, tuple)):
            raise TypeError(f"img_size必须是int或tuple类型，但得到了{type(img_size)}")
        if not isinstance(patch_size, (int, tuple)):
            raise TypeError(f"patch_size必须是int或tuple类型，但得到了{type(patch_size)}")
        if not isinstance(in_channels, int):
            raise TypeError(f"in_channels必须是int类型，但得到了{type(in_channels)}")
        if not isinstance(embed_dim, int):
            raise TypeError(f"embed_dim必须是int类型，但得到了{type(embed_dim)}")
            
        # 转换为元组格式
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 检查参数值的有效性
        if self.img_size[0] <= 0 or self.img_size[1] <= 0:
            raise ValueError(f"图像尺寸必须大于0，但得到了{self.img_size}")
        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError(f"patch尺寸必须大于0，但得到了{self.patch_size}")
        if self.in_channels <= 0:
            raise ValueError(f"输入通道数必须大于0，但得到了{self.in_channels}")
        if self.embed_dim <= 0:
            raise ValueError(f"嵌入维度必须大于0，但得到了{self.embed_dim}")
        
        # 检查图像尺寸和patch尺寸是否合适
        if self.img_size[0] % self.patch_size[0] != 0 or self.img_size[1] % self.patch_size[1] != 0:
            print(f"警告: 图像尺寸 {self.img_size} 不能被patch尺寸 {self.patch_size} 整除")
            # 调整图像尺寸为能被patch_size整除的最近值
            new_h = (self.img_size[0] // self.patch_size[0]) * self.patch_size[0]
            new_w = (self.img_size[1] // self.patch_size[1]) * self.patch_size[1]
            print(f"将调整图像尺寸为: ({new_h}, {new_w})")
            self.img_size = (new_h, new_w)
        
        # 计算总的patch数量（适应任意形状的图像）
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # 使用卷积实现patch嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # 初始化权重
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        
        # 调试标志
        self.debug_forward = False
        
        # 打印初始化信息
        print(f"\n[PatchEmbed] 初始化完成:")
        print(f"图像尺寸: {self.img_size}")
        print(f"Patch尺寸: {self.patch_size}")
        print(f"输入通道数: {self.in_channels}")
        print(f"嵌入维度: {self.embed_dim}")
        print(f"网格尺寸: {self.grid_size}")
        print(f"Patch数量: {self.num_patches}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        Returns:
            patch嵌入向量，形状为 [batch_size, num_patches, embed_dim]
        """
        # 检查输入数据的有效性
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"输入必须是torch.Tensor类型，但得到了{type(x)}")
            
        if x.dim() != 4:
            raise ValueError(f"输入必须是4维张量 [batch_size, channels, height, width]，但形状是{x.shape}")
            
        if x.size(1) != self.in_channels:
            raise ValueError(f"输入通道数必须是{self.in_channels}，但得到了{x.size(1)}")
            
        B, C, H, W = x.shape
        
        if self.debug_forward:
            print(f"[PatchEmbed] 输入: 形状={x.shape}, 设备={x.device}")
        
        # 检查输入尺寸并调整大小
        if H != self.img_size[0] or W != self.img_size[1]:
            if self.debug_forward:
                print(f"[PatchEmbed] 调整输入尺寸从 ({H}, {W}) 到 {self.img_size}")
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
            if self.debug_forward:
                print(f"[PatchEmbed] 调整后的形状: {x.shape}")
        
        # 应用卷积进行分块和投影
        x = self.proj(x)  # [B, embed_dim, grid_h, grid_w]
        
        if self.debug_forward:
            print(f"\n投影后:")
            print(f"形状: {x.shape}")
            print(f"数据范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"数据类型: {x.dtype}")
            print(f"设备: {x.device}")
        
        # 检查输出尺寸
        _, E, gH, gW = x.shape
        if gH != self.grid_size[0] or gW != self.grid_size[1]:
            raise ValueError(
                f"输出网格尺寸 ({gH}, {gW}) 与预期网格尺寸 {self.grid_size} 不匹配"
            )
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2).reshape(x.size(0), -1, self.embed_dim)
        
        if self.debug_forward:
            print(f"\n最终输出:")
            print(f"形状: {x.shape}")
            print(f"数据范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
            print(f"数据类型: {x.dtype}")
            print(f"设备: {x.device}")
            print(f"是否与预期patch数量匹配: {x.size(1) == self.num_patches}")
        
        return x

class AttentiveRouter(nn.Module):
    """注意力路由器 - 只对一般专家进行路由选择"""
    def __init__(
        self,
        input_dim: int,
        num_general_experts: int,  # 一般专家的数量
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_general_experts = num_general_experts
        self.top_k = top_k
        
        # 路由器只针对一般专家
        self.router = nn.Linear(input_dim, num_general_experts)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数，使用特殊初始化"""
        nn.init.orthogonal_(self.router.weight, gain=0.1)
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: [batch_size, seq_len, input_dim]
        Returns:
            包含路由决策的字典：
            - expert_weights: [batch_size, seq_len, top_k]
            - expert_indices: [batch_size, seq_len, top_k]
        """
        # 计算一般专家的路由logits
        router_logits = self.router(inputs)
        
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 选择top-k个一般专家
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 重新归一化权重
        top_k_weights = F.softmax(top_k_probs, dim=-1)
        
        return {
            'logits': router_logits,
            'router_probs': router_probs,
            'expert_weights': top_k_weights,
            'expert_indices': top_k_indices
        }

class UnifiedModalEncoder(nn.Module):
    """MoE块 - 包含11个并行专家"""
    def __init__(
        self,
        embed_dim: int,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # 1. 全局共享专家 - 处理所有token
        self.global_expert = Expert(
                n_embd=embed_dim,
                expansion_factor=4,
                dropout=dropout,
                activation=activation
            )
        
        # 2. 模态特定专家
        self.vision_expert = Expert(
                n_embd=embed_dim,
                expansion_factor=4,
                dropout=dropout,
                activation=activation
            )
        self.text_expert = Expert(
            n_embd=embed_dim,
            expansion_factor=4,
            dropout=dropout,
            activation=activation
        )
        
        # 3. 一般专家
        self.general_experts = nn.ModuleList([
            Expert(
                n_embd=embed_dim,
                expansion_factor=4,
            dropout=dropout,
                activation=activation
        )
            for _ in range(num_general_experts)
        ])
        
        # 路由器 - 只针对一般专家
        self.router = AttentiveRouter(
            input_dim=embed_dim,
            num_general_experts=num_general_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm层
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 配置
        self.use_checkpoint = use_checkpoint
        self.top_k = top_k
        self.num_general_experts = num_general_experts
        
    def forward(self, x: torch.Tensor, modality_type: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            x: [batch_size, seq_len, embed_dim]
            modality_type: [batch_size, seq_len] 0=图像, 1=文本, 0.5=融合层
        """
        # 如果未提供模态类型，默认为全0（图像）
        if modality_type is None:
            modality_type = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            
        # 保存残差
        residual = x
        
        # 1. 自注意力
        x = self.norm1(x)
        attn_output, attn_weights = self.attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # 保存残差
        residual = x
        
        # 2. MoE层
        x = self.norm2(x)
        
        # 初始化专家输出
        batch_size, seq_len, embed_dim = x.shape
        expert_outputs = torch.zeros_like(x)
        
        # 全局共享专家 - 处理所有token
        global_output = self.global_expert(x)
        expert_outputs += global_output
        
        # 模态特定专家 - 根据模态类型处理token
        # 只有非融合层(即模态类型不为0.5)才使用模态特定专家
        is_fusion_layer = (modality_type == 0.5).any()
        
        if not is_fusion_layer:
            vision_mask = (modality_type == 0)
            text_mask = (modality_type == 1)
            
            if vision_mask.any():
                vision_tokens = x[vision_mask]
                vision_output = self.vision_expert(vision_tokens)
                expert_outputs[vision_mask] += vision_output
                
            if text_mask.any():
                text_tokens = x[text_mask]
                text_output = self.text_expert(text_tokens)
                expert_outputs[text_mask] += text_output
        
        # 一般专家 - 通过路由选择处理token
        router_outputs = self.router(x)
        expert_indices = router_outputs['expert_indices']  # [batch_size, seq_len, top_k]
        expert_weights = router_outputs['expert_weights']  # [batch_size, seq_len, top_k]
        
        # 处理每个token的top-k个专家
        for k in range(self.top_k):
            # 获取当前k位置的专家索引和权重
            k_indices = expert_indices[:, :, k]  # [batch_size, seq_len]
            k_weights = expert_weights[:, :, k].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # 处理每个一般专家
            for expert_idx in range(self.num_general_experts):
                # 找到路由到当前专家的token
                mask = (k_indices == expert_idx)
                if mask.any():
                    # 提取相关token
                    tokens = x[mask]
                    # 通过专家处理
                    output = self.general_experts[expert_idx](tokens)
                    # 加权累加到输出中
                    expert_outputs[mask] += output * k_weights[mask]
        
        # 应用残差连接
        x = residual + self.dropout(expert_outputs)
        
        return {
            'output': x,
            'router_logits': router_outputs.get('logits', None),
            'router_probs': router_outputs.get('router_probs', None),
            'attention_weights': attn_weights
        }

class ImageEncoder(nn.Module):
    """图像MoE编码器 - 由6个MoE块组成"""
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 6,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # 创建6个MoE块
        self.layers = nn.ModuleList([
            UnifiedModalEncoder(
                embed_dim=embed_dim,
                num_general_experts=num_general_experts,
                top_k=top_k,
                dropout=dropout,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            )
            for _ in range(num_layers)
        ])
        
        # 最终Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        # 指定为图像模态
        modality_type = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        
        # 存储各层输出
        layer_outputs = {}
        
        # 通过各层处理
        for i, layer in enumerate(self.layers):
            outputs = layer(x, modality_type)
            x = outputs['output']
            layer_outputs[f'layer_{i}'] = outputs
        
        # 最终归一化
        x = self.norm(x)
        
        return {
            'output': x,
            'layer_outputs': layer_outputs
        }

class TextEncoder(nn.Module):
    """文本MoE编码器 - 由4个MoE块组成"""
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # 创建4个MoE块
        self.layers = nn.ModuleList([
            UnifiedModalEncoder(
                embed_dim=embed_dim,
                num_general_experts=num_general_experts,
                top_k=top_k,
                dropout=dropout,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            )
            for _ in range(num_layers)
        ])
        
        # 最终Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        # 指定为文本模态
        modality_type = torch.ones(x.shape[0], x.shape[1], device=x.device)
        
        # 存储各层输出
        layer_outputs = {}
        
        # 通过各层处理
        for i, layer in enumerate(self.layers):
            outputs = layer(x, modality_type)
            x = outputs['output']
            layer_outputs[f'layer_{i}'] = outputs
        
        # 最终归一化
        x = self.norm(x)
        
        return {
            'output': x,
            'layer_outputs': layer_outputs
        }

class CrossModalFusion(nn.Module):
    """跨模态融合层 - 由3个MoE块组成"""
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        use_checkpoint: bool = False
    ):
        super().__init__()
        
        # 创建3个MoE块
        self.layers = nn.ModuleList([
            UnifiedModalEncoder(
                embed_dim=embed_dim,
                num_general_experts=num_general_experts,
                top_k=top_k,
                dropout=dropout,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            )
            for _ in range(num_layers)
        ])
        
        # 跨模态注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 最终Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            visual_features: [batch_size, vis_seq_len, embed_dim]
            text_features: [batch_size, text_seq_len, embed_dim]
        """
        # 应用跨模态注意力
        residual = visual_features
        visual_features = self.norm1(visual_features)
        
        # 图像特征关注文本特征
        cross_out, cross_weights = self.cross_attention(
            query=visual_features, 
            key=text_features, 
            value=text_features
        )
        
        # 门控融合
        gate_value = self.gate(cross_out)
        fused_features = gate_value * cross_out + (1 - gate_value) * residual
        
        # 设置混合模态类型 (0.5表示混合模态)
        batch_size, seq_len, embed_dim = fused_features.shape
        modality_type = torch.ones(batch_size, seq_len, device=fused_features.device) * 0.5
        
        # 存储各层输出
        layer_outputs = {}
        
        # 通过MoE块处理
        x = fused_features
        for i, layer in enumerate(self.layers):
            outputs = layer(x, modality_type)
            x = outputs['output']
            layer_outputs[f'fusion_layer_{i}'] = outputs
        
        # 最终归一化
        x = self.norm(x)
        
        return {
            'output': x,
            'layer_outputs': layer_outputs,
            'cross_attention_weights': cross_weights,
            'gate_value': gate_value
        }

class MultiModalMoE(nn.Module):
    """多模态混合专家模型 - 完整架构"""
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 512,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        img_encoder_layers: int = 6,
        text_encoder_layers: int = 4,
        fusion_layers: int = 3,
        device: str = 'cuda',
        vocab_size: int = 50000,
        max_text_len: int = 32,
        text_embed_dim: int = 128,
        use_checkpoint: bool = False
    ):
        """初始化多模态MoE模型"""
        super().__init__()
        
        self.device = device
        
        # 保存配置
        self.config = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'embed_dim': embed_dim,
            'num_general_experts': num_general_experts,
            'top_k': top_k,
            'dropout': dropout,
            'num_heads': num_heads,
            'img_encoder_layers': img_encoder_layers,
            'text_encoder_layers': text_encoder_layers,
            'fusion_layers': fusion_layers,
            'device': device,
            'vocab_size': vocab_size,
            'max_text_len': max_text_len,
            'text_embed_dim': text_embed_dim,
            'use_checkpoint': use_checkpoint,
            'initializer_range': 0.02
        }
        
        # 图像Patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # 计算patch序列长度
        self.seq_length = self.patch_embed.num_patches
        
        # 图像位置嵌入
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        
        # 添加模态类型嵌入
        self.token_type_embed = nn.Embedding(2, embed_dim)  # 0=视觉, 1=文本
        
        # 文本嵌入层
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim)
        
        # 文本位置嵌入
        max_pos_len = max(max_text_len, 77)
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_pos_len, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 图像编码器
        self.image_encoder = ImageEncoder(
                embed_dim=embed_dim,
            num_layers=img_encoder_layers,
            num_general_experts=num_general_experts,
                top_k=top_k,
                dropout=dropout,
                num_heads=num_heads,
            use_checkpoint=use_checkpoint
        )
        
        # 文本编码器
        self.text_encoder = TextEncoder(
            embed_dim=embed_dim,
            num_layers=text_encoder_layers,
            num_general_experts=num_general_experts,
            top_k=top_k,
            dropout=dropout,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint
        )
        
        # 跨模态融合
        self.cross_modal_fusion = CrossModalFusion(
            embed_dim=embed_dim,
            num_layers=fusion_layers,
            num_general_experts=num_general_experts,
            top_k=top_k,
            dropout=dropout,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint
        )
        
        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 初始化
        self.apply(self._init_weights)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.img_pos_embed, std=self.config['initializer_range'])
        nn.init.trunc_normal_(self.text_pos_embed, std=self.config['initializer_range'])
        
        # 损失函数权重
        self.router_z_loss_weight = 0.001  # 路由器正则化损失权重
        self.router_balance_loss_weight = 0.01  # 路由器平衡损失权重
        self.cross_modal_alignment_weight = 0.1  # 跨模态对齐损失权重
        self.contrastive_loss_weight = 0.1  # 对比损失权重
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def interpolate_pos_encoding(self, x: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """对位置编码进行插值，使其适应不同大小的输入"""
        _, seq_len, dim = x.shape
        _, orig_seq_len, _ = pos_embed.shape
        
        if seq_len == orig_seq_len:
            return pos_embed
            
        orig_size = int(math.sqrt(orig_seq_len))
        new_size = int(math.sqrt(seq_len))
        
        if orig_size == new_size:
            return pos_embed
            
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, 
            size=(new_size, new_size), 
            mode='bicubic', 
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)
        return pos_embed
        
    def forward(self, 
                images: torch.Tensor, 
                text_tokens: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            images: [batch_size, in_channels, height, width]
            text_tokens: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 处理图像输入
        # 图像Patch嵌入
        img_tokens = self.patch_embed(images)
            
            # 应用位置编码
        if img_tokens.size(1) != self.img_pos_embed.size(1):
            pos_embed = self.interpolate_pos_encoding(img_tokens, self.img_pos_embed)
        else:
            pos_embed = self.img_pos_embed
            
            # 确保位置编码在正确的设备上
            pos_embed = pos_embed.to(device)
            
            # 加上位置编码
        img_tokens = img_tokens + pos_embed
            
            # 添加类型编码（图像token为0）
        img_type_ids = torch.zeros(batch_size, img_tokens.size(1), dtype=torch.long, device=device)
        img_tokens = img_tokens + self.token_type_embed(img_type_ids)
            
            # 应用dropout
        img_tokens = self.dropout(img_tokens)
            
        # 2. 处理文本输入
        text_tokens_processed = None
        text_features = None
        if text_tokens is not None:
            # 确保文本输入在正确的设备上
            text_tokens = text_tokens.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            # 文本嵌入
            text_features = self.text_embedding(text_tokens)
            text_features = self.text_projection(text_features)
                
            # 添加文本位置编码
            text_pos_embed = self.text_pos_embed[:, :text_features.size(1), :].to(device)
            text_features = text_features + text_pos_embed
                
            # 添加类型编码（文本token为1）
            text_type_ids = torch.ones(batch_size, text_features.size(1), dtype=torch.long, device=device)
            text_features = text_features + self.token_type_embed(text_type_ids)
                
            # 应用dropout
            text_tokens_processed = self.dropout(text_features)
        
        # 3. 通过编码器处理图像和文本
        img_encoder_outputs = self.image_encoder(img_tokens)
        img_features = img_encoder_outputs['output']
        
        # 如果有文本输入，处理文本并进行跨模态融合
        fusion_outputs = {}
        if text_tokens_processed is not None:
            text_encoder_outputs = self.text_encoder(text_tokens_processed)
            text_features = text_encoder_outputs['output']
            
            # 跨模态融合
            fusion_outputs = self.cross_modal_fusion(img_features, text_features)
            final_features = fusion_outputs['output']
        else:
            # 如果没有文本输入，只使用图像特征
            final_features = img_features
            fusion_outputs = {'output': final_features}
        
        # 4. 全局池化 - 平均池化
        pooled_features = final_features.mean(dim=1)
        
        # 5. 分类
        logits = self.classifier(pooled_features)
        
        # 收集所有层的路由决策，用于计算路由损失
        all_router_logits = []
        all_router_probs = []
        
        # 收集图像编码器路由决策
        for i in range(self.config['img_encoder_layers']):
            layer_outputs = img_encoder_outputs['layer_outputs'][f'layer_{i}']
            if 'router_logits' in layer_outputs and layer_outputs['router_logits'] is not None:
                all_router_logits.append(layer_outputs['router_logits'])
            if 'router_probs' in layer_outputs and layer_outputs['router_probs'] is not None:
                all_router_probs.append(layer_outputs['router_probs'])
        
        # 如果有文本输入，收集文本编码器和融合层路由决策
        if text_tokens_processed is not None:
            # 文本编码器路由决策
            for i in range(self.config['text_encoder_layers']):
                layer_outputs = text_encoder_outputs['layer_outputs'][f'layer_{i}']
                if 'router_logits' in layer_outputs and layer_outputs['router_logits'] is not None:
                    all_router_logits.append(layer_outputs['router_logits'])
                if 'router_probs' in layer_outputs and layer_outputs['router_probs'] is not None:
                    all_router_probs.append(layer_outputs['router_probs'])
            
            # 融合层路由决策
            for i in range(self.config['fusion_layers']):
                layer_outputs = fusion_outputs['layer_outputs'][f'fusion_layer_{i}']
                if 'router_logits' in layer_outputs and layer_outputs['router_logits'] is not None:
                    all_router_logits.append(layer_outputs['router_logits'])
                if 'router_probs' in layer_outputs and layer_outputs['router_probs'] is not None:
                    all_router_probs.append(layer_outputs['router_probs'])
        
        # 计算路由损失
        router_z_loss = torch.tensor(0.0, device=device)
        router_balance_loss = torch.tensor(0.0, device=device)
        
        for router_logits in all_router_logits:
            router_z_loss = router_z_loss + self.compute_z_loss(router_logits)
            
        for router_probs in all_router_probs:
            router_balance_loss = router_balance_loss + self.compute_load_loss(router_probs)
        
        # 计算跨模态对齐损失
        cross_modal_loss = torch.tensor(0.0, device=device)
        contrastive_loss = torch.tensor(0.0, device=device)
        if text_tokens_processed is not None:
            # 简单的余弦相似度损失
            img_mean = img_features.mean(dim=1)
            text_mean = text_features.mean(dim=1)
            
            img_norm = F.normalize(img_mean, p=2, dim=1)
            text_norm = F.normalize(text_mean, p=2, dim=1)
            
            # 最大化余弦相似度（最小化负相似度）
            cross_modal_loss = -torch.sum(img_norm * text_norm) / batch_size
            
            # 计算对比损失
            # 计算所有样本对之间的相似度矩阵
            similarity_matrix = torch.matmul(img_norm, text_norm.transpose(0, 1))
            
            # 正样本对（对角线）和负样本对
            labels = torch.arange(batch_size, device=device)
            
            # 计算对比损失（InfoNCE损失）
            contrastive_loss_i2t = F.cross_entropy(similarity_matrix / 0.07, labels)
            contrastive_loss_t2i = F.cross_entropy(similarity_matrix.t() / 0.07, labels)
            contrastive_loss = (contrastive_loss_i2t + contrastive_loss_t2i) / 2.0
        
        # 总路由损失
        router_loss = (
            self.router_z_loss_weight * router_z_loss + 
            self.router_balance_loss_weight * router_balance_loss +
            self.cross_modal_alignment_weight * cross_modal_loss +
            self.contrastive_loss_weight * contrastive_loss
        )
        
        return {
            'logits': logits,
            'embeddings': pooled_features,
            'img_features': img_features,
            'text_features': text_features if text_tokens_processed is not None else None,
            'fused_features': final_features,
            'router_z_loss': router_z_loss,
            'router_balance_loss': router_balance_loss,
            'cross_modal_loss': cross_modal_loss,
            'contrastive_loss': contrastive_loss,
            'router_loss': router_loss,
            # 存储专家激活情况，用于可视化
            'expert_activations': {
                'img_encoder': img_encoder_outputs.get('layer_outputs', {}),
                'text_encoder': text_encoder_outputs.get('layer_outputs', {}) if text_tokens_processed is not None else {},
                'fusion': fusion_outputs.get('layer_outputs', {}) if text_tokens_processed is not None else {}
            }
        }
    
    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算z损失来正则化路由逻辑"""
        # 计算每个专家的平均路由概率
        router_probs = torch.softmax(router_logits, dim=-1)
        # 计算router_z的平方（用于正则化）
        mean_probs = router_probs.mean(dim=(0, 1))
        router_z = torch.mean(mean_probs ** 2) * router_probs.shape[-1]
        return router_z

    def compute_load_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """计算负载平衡损失，以确保专家的使用均衡"""
        # 获取专家数量
        num_experts = router_probs.shape[-1]
        
        # 计算每个专家的使用频率
        expert_usage = router_probs.mean(dim=(0, 1))
        
        # 计算理想的均匀分布
        ideal_usage = torch.ones_like(expert_usage) / num_experts
        
        # 使用KL散度计算与理想分布的差异
        load_loss = torch.sum(expert_usage * torch.log(expert_usage / ideal_usage + 1e-9))
        return load_loss

class ModelWrapper(nn.Module):
    """统一模型接口"""
    def __init__(self, model: nn.Module, preprocess: transforms.Compose):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        
    def forward(self, x: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """前向传播函数"""
        return self.model(x, *args, **kwargs)
        
    @torch.inference_mode()
    def predict(self, inputs: Union[np.ndarray, List[Image.Image]]) -> Dict[str, Any]:
        """统一预测接口"""
        # 转换输入格式
        if isinstance(inputs, list):
            inputs = [self.preprocess(img) for img in inputs]
            inputs = torch.stack(inputs)
        elif isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
            
        # 执行预测
        outputs = self.model(inputs)
        return {
            'logits': outputs['logits'],
            'attention_weights': outputs.get('attention_weights', None),
            'router_decisions': outputs.get('router_outputs', None)
        } 