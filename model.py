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
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        
        # 计算总的patch数量（适应任意形状的图像）
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # 使用卷积实现patch嵌入
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像张量，形状为 [batch_size, in_channels, height, width]
            
        Returns:
            patch嵌入向量，形状为 [batch_size, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        # 这里不做强制断言，允许输入不同大小的图像
        # 让模型自适应处理不同尺寸的输入
        
        # 应用卷积进行分块和投影
        x = self.proj(x)  # [B, embed_dim, grid_h, grid_w]
        
        # 重塑为序列
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        return x

class AttentiveRouter(nn.Module):
    """注意力路由器
    
    基于注意力机制的路由器，将输入分配给多个专家
    """
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        noisy_routing: bool = False,
        use_softmax: bool = True,
        dropout: float = 0.1
    ):
        """初始化
        
        Args:
            input_dim: 输入维度
            num_experts: 专家数量
            top_k: 每个token选择的专家数量
            capacity_factor: 容量因子
            noisy_routing: 是否使用噪声路由
            use_softmax: 是否使用softmax（否则使用sigmoid）
            dropout: Dropout比率
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_routing = noisy_routing
        self.use_softmax = use_softmax
        self.noise_std = 0.01  # 噪声标准差
        
        # 路由器 - 映射输入到专家选择
        self.router = nn.Linear(input_dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数，使用特殊初始化"""
        # 使用正交初始化
        nn.init.orthogonal_(self.router.weight, gain=0.1)
            
    def forward(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """前向传播
        
        Args:
            inputs: 输入张量 [batch_size, seq_len, input_dim]
            
        Returns:
            包含路由决策的字典：
                - logits: 路由逻辑 [batch_size, seq_len, num_experts]
                - router_probs: 路由概率 [batch_size, seq_len, num_experts]
                - expert_weights: 专家权重 [batch_size, seq_len, top_k]
                - expert_indices: 专家索引 [batch_size, seq_len, top_k]
        """
        batch_size, seq_len, input_dim = inputs.shape
        
        # 计算路由logits: [batch_size, seq_len, num_experts]
        router_logits = self.router(inputs)
        
        # 添加可选的训练噪声
        if self.noisy_routing and self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std
        
        # 计算每个token的top_k专家
        if self.use_softmax:
            # Softmax版本 - 使用softmax进行归一化
            router_probs = F.softmax(router_logits, dim=-1)
        else:
            # Sigmoid版本 - 直接输出每个专家的重要性（可能值不是和为1）
            router_probs = torch.sigmoid(router_logits)
        
        # 获取top_k专家及其权重
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # 重新归一化top_k专家权重，使其和为1
        if self.use_softmax:
            # 对于softmax，直接除以sum即可
            top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        else:
            # 对于sigmoid，使用softmax重新归一化
            top_k_weights = F.softmax(top_k_probs, dim=-1)
        
        return {
            'logits': router_logits,              # [batch_size, seq_len, num_experts]
            'router_probs': router_probs,         # [batch_size, seq_len, num_experts]
            'expert_weights': top_k_weights,      # [batch_size, seq_len, top_k]
            'expert_indices': top_k_indices       # [batch_size, seq_len, top_k]
        }

class UnifiedModalEncoder(nn.Module):
    """统一的模态编码器"""
    def __init__(
        self,
        embed_dim: int,
        num_shared_experts: int,
        num_modality_specific_experts: int,
        top_k: int,
        dropout: float,
        num_heads: int,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        use_bias: bool = True,
        use_checkpoint: bool = False
    ):
        """初始化
        
        Args:
            embed_dim: 嵌入维度
            num_shared_experts: 共享专家数量
            num_modality_specific_experts: 模态特定专家数量
            top_k: 每个token选择的专家数量
            dropout: Dropout比率
            num_heads: 注意力头数量
            activation: 激活函数 ('gelu' 或 'relu')
            layer_norm_eps: Layer Norm的epsilon值
            use_bias: 是否使用偏置
            use_checkpoint: 是否使用梯度检查点
        """
        super().__init__()
        
        # 保存配置
        self.embed_dim = embed_dim
        self.num_shared_experts = num_shared_experts
        self.num_modality_specific_experts = num_modality_specific_experts
        self.top_k = top_k
        self.dropout = dropout
        self.num_heads = num_heads
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.use_bias = use_bias
        self.use_checkpoint = use_checkpoint
        
        # Layer Norm层
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.cross_modal_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 创建专家
        # 共享专家 - 对所有token都可用
        self.shared_experts = nn.ModuleList([
            Expert(
                n_embd=embed_dim,
                expansion_factor=4,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_shared_experts)
        ])
        
        # 模态特定专家 - 仅对特定模态token可用
        self.modality_specific_experts = nn.ModuleList([
            Expert(
                n_embd=embed_dim,
                expansion_factor=4,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_modality_specific_experts)
        ])
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨模态注意力层
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 路由器 - 决定每个token使用哪些专家
        total_experts = num_shared_experts + num_modality_specific_experts
        self.router = AttentiveRouter(
            input_dim=embed_dim,
            num_experts=total_experts,
            top_k=top_k,
            dropout=dropout
        )
        
        # 门控机制 - 用于控制跨模态融合
        self.modal_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
        
        # 初始化parent_model属性
        self.parent_model = None
        
    def _init_weights(self):
        """初始化模型权重"""
        # 使用更通用的初始化方法，遍历所有模块
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 初始化线性层
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # 初始化LayerNorm
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # 初始化注意力层
                if hasattr(module, 'in_proj_weight'):
                    nn.init.normal_(module.in_proj_weight, std=0.02)
                if hasattr(module, 'out_proj'):
                    nn.init.normal_(module.out_proj.weight, std=0.02)
                    nn.init.zeros_(module.out_proj.bias)
        
        # 初始化专家路由器
        if hasattr(self, 'router'):
            self.router.reset_parameters()
        
        # 初始化current_labels属性
        self.current_labels = None
        self.debug_cross_modal = False
        
    def _attention_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """自注意力块，用于处理序列信息
        
        Args:
            x: 输入特征 [batch_size, seq_len, embed_dim]
            
        Returns:
            tuple: (注意力输出, 注意力权重)
        """
        # 多头自注意力
        attn_output, attn_weights = self.attention(
            query=x,
            key=x,
            value=x
        )
        
        return attn_output, attn_weights
    
    def _cross_modal_fusion(self, x: torch.Tensor, modal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """跨模态融合层，用于整合不同模态的信息"""
        if modal_context is None:
            return x
            
        residual = x
        x = self.cross_modal_norm(x)
        
        # 添加形状检查和调试信息
        batch_size, seq_len, embed_dim = x.shape
        modal_batch, modal_seq, modal_dim = modal_context.shape
        
        # 调试打印 - 仅在设置了debug标志时输出
        if hasattr(self, 'debug_cross_modal') and self.debug_cross_modal:
            print(f"\n[跨模态融合] 调试信息:")
            print(f"  图像特征形状: {x.shape} - [批次大小, 序列长度, 嵌入维度]")
            print(f"  文本特征形状: {modal_context.shape} - [批次大小, 序列长度, 嵌入维度]")
            
            # 打印modal_context的一些统计信息，帮助判断是否包含有效内容
            if modal_context is not None:
                modal_mean = modal_context.mean().item()
                modal_std = modal_context.std().item()
                modal_min = modal_context.min().item()
                modal_max = modal_context.max().item()
                print(f"  文本特征统计: 均值={modal_mean:.4f}, 标准差={modal_std:.4f}, 最小值={modal_min:.4f}, 最大值={modal_max:.4f}")
        
        # 确保维度匹配
        if embed_dim != modal_dim:
            raise ValueError(f"嵌入维度不匹配: x={embed_dim}, modal_context={modal_dim}")
        
        # 处理批次大小不匹配的情况
        if modal_batch != batch_size:
            error_msg = f"批次大小不匹配: 图像批次大小={batch_size}, 文本特征批次大小={modal_batch}"
            if hasattr(self, 'current_labels'):
                error_msg += f"\n当前标签: {self.current_labels if self.current_labels is not None else 'None'}"
            error_msg += "\n解决方法:\n1. 确保文本特征的批次大小与图像批次大小相同\n2. 在调用forward前使用model.set_labels(labels)设置当前批次的标签"
            raise ValueError(error_msg)
        
        # 使用注意力机制进行跨模态融合
        try:
            # 如果modal_context只有一个序列元素，尝试扩展它以匹配x的序列长度
            if modal_seq == 1:
                modal_context_expanded = modal_context.expand(modal_batch, seq_len, modal_dim)
                fusion_output, _ = self.cross_modal_attention(x, modal_context_expanded, modal_context_expanded)
                
                # 打印调试信息
                if hasattr(self, 'debug_cross_modal') and self.debug_cross_modal:
                    print(f"  注意: 文本序列长度为1，已扩展到 {seq_len} 以匹配图像序列长度")
            else:
                fusion_output, _ = self.cross_modal_attention(x, modal_context, modal_context)
                
                # 打印调试信息
                if hasattr(self, 'debug_cross_modal') and self.debug_cross_modal:
                    print(f"  直接进行注意力融合，无需扩展文本特征")
                    
            # 打印融合输出的统计信息
            if hasattr(self, 'debug_cross_modal') and self.debug_cross_modal:
                fusion_mean = fusion_output.mean().item()
                fusion_std = fusion_output.std().item()
                print(f"  融合输出统计: 形状={fusion_output.shape}, 均值={fusion_mean:.4f}, 标准差={fusion_std:.4f}")
                
        except RuntimeError as e:
            # 打印详细的形状信息以便调试
            print(f"跨模态融合错误: x形状={x.shape}, modal_context形状={modal_context.shape}")
            print(f"x数据类型={x.dtype}, modal_context数据类型={modal_context.dtype}")
            print(f"x设备={x.device}, modal_context设备={modal_context.device}")
            # 重新抛出异常
            raise RuntimeError(f"跨模态融合失败: {str(e)}")
        
        # 使用门控机制控制融合程度
        gate = self.modal_gate(fusion_output)
        fusion_output = gate * fusion_output + (1 - gate) * residual
        
        # 打印门控值的统计信息
        if hasattr(self, 'debug_cross_modal') and self.debug_cross_modal:
            gate_mean = gate.mean().item()
            gate_std = gate.std().item()
            print(f"  门控值统计: 均值={gate_mean:.4f}, 标准差={gate_std:.4f}\n")
            
            # 打印完后关闭调试模式，以免输出过多日志
            self.debug_cross_modal = False
        
        return fusion_output
    
    def _router_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mixture of Experts路由层
        
        Args:
            x: 输入特征 [batch_size, seq_len, embed_dim]
            
        Returns:
            tuple: (输出特征, 路由逻辑, 路由概率)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # 路由决策 - 决定每个token应该送到哪些专家
        # [batch_size, seq_len, num_experts]
        router_outputs = self.router(x)
        
        # 获取路由logits和概率
        router_logits = router_outputs['logits']  # [batch_size, seq_len, num_experts]
        router_probs = router_outputs['router_probs']  # [batch_size, seq_len, num_experts]
        
        # 获取最终的专家分配和组合权重
        # 每个token会被分配到top_k个专家
        expert_weights = router_outputs['expert_weights']  # [batch_size, seq_len, top_k]
        expert_indices = router_outputs['expert_indices']  # [batch_size, seq_len, top_k]
        
        # 初始化输出tensor
        final_output = torch.zeros_like(x)  # [batch_size, seq_len, embed_dim]
        
        # 展平输入以方便处理
        flat_x = x.reshape(-1, embed_dim)  # [batch_size * seq_len, embed_dim]
        
        # 准备索引用于收集每个token对应的专家输出
        # 创建token索引
        token_indices = torch.arange(batch_size * seq_len, device=x.device)
        
        # 处理共享专家
        shared_outputs = []
        for i, expert in enumerate(self.shared_experts):
            # 找出被路由到当前专家的所有token
            expert_mask = (expert_indices == i)  # [batch_size, seq_len, top_k]
            if not expert_mask.any():
                continue
                
            # 展平mask
            flat_mask = expert_mask.reshape(-1, self.top_k)  # [batch_size * seq_len, top_k]
            
            # 找出至少有一个专家被路由到当前专家的token
            token_mask = flat_mask.any(dim=-1)  # [batch_size * seq_len]
            
            if not token_mask.any():
                continue
                
            # 选择这些token
            selected_tokens = token_indices[token_mask]  # [num_selected]
            selected_inputs = flat_x[selected_tokens]  # [num_selected, embed_dim]
            
            # 应用专家
            expert_output = expert(selected_inputs)  # [num_selected, embed_dim]
            
            # 收集每个被路由到当前专家的token对（token_idx, expert_idx）
            # 并计算其组合权重
            for k in range(self.top_k):
                # 当前专家索引在k位置的mask
                k_mask = expert_mask[:, :, k].reshape(-1)  # [batch_size * seq_len]
                
                if not k_mask.any():
                    continue
                    
                # 选择这些token
                k_tokens = token_indices[k_mask]  # [num_k_selected]
                
                # 获取权重
                k_weights = expert_weights[:, :, k].reshape(-1)[k_mask]  # [num_k_selected]
                
                # 获取这些token在被选中的token中的索引
                k_selected_indices = torch.zeros_like(token_mask, dtype=torch.bool)
                k_selected_indices[k_tokens] = True
                k_selected_indices = k_selected_indices & token_mask  # 确保只选择被路由到当前专家的token
                k_indices_in_selected = torch.nonzero(k_selected_indices).squeeze(-1)  # [num_k_selected]
                
                # 加权求和
                final_output.reshape(-1, embed_dim)[k_tokens] += k_weights.unsqueeze(-1) * expert_output[k_indices_in_selected]
        
        # 处理模态特定专家
        modality_specific_outputs = []
        for i, expert in enumerate(self.modality_specific_experts):
            # 调整专家索引（共享专家之后）
            expert_idx = i + len(self.shared_experts)
            
            # 找出被路由到当前专家的所有token
            expert_mask = (expert_indices == expert_idx)  # [batch_size, seq_len, top_k]
            if not expert_mask.any():
                continue
                
            # 展平mask
            flat_mask = expert_mask.reshape(-1, self.top_k)  # [batch_size * seq_len, top_k]
            
            # 找出至少有一个专家被路由到当前专家的token
            token_mask = flat_mask.any(dim=-1)  # [batch_size * seq_len]
            
            if not token_mask.any():
                continue
                
            # 选择这些token
            selected_tokens = token_indices[token_mask]  # [num_selected]
            selected_inputs = flat_x[selected_tokens]  # [num_selected, embed_dim]
            
            # 应用专家
            expert_output = expert(selected_inputs)  # [num_selected, embed_dim]
            
            # 收集每个被路由到当前专家的token对（token_idx, expert_idx）
            # 并计算其组合权重
            for k in range(self.top_k):
                # 当前专家索引在k位置的mask
                k_mask = expert_mask[:, :, k].reshape(-1)  # [batch_size * seq_len]
                
                if not k_mask.any():
                    continue
                    
                # 选择这些token
                k_tokens = token_indices[k_mask]  # [num_k_selected]
                
                # 获取权重
                k_weights = expert_weights[:, :, k].reshape(-1)[k_mask]  # [num_k_selected]
                
                # 获取这些token在被选中的token中的索引
                k_selected_indices = torch.zeros_like(token_mask, dtype=torch.bool)
                k_selected_indices[k_tokens] = True
                k_selected_indices = k_selected_indices & token_mask  # 确保只选择被路由到当前专家的token
                k_indices_in_selected = torch.nonzero(k_selected_indices).squeeze(-1)  # [num_k_selected]
                
                # 加权求和
                final_output.reshape(-1, embed_dim)[k_tokens] += k_weights.unsqueeze(-1) * expert_output[k_indices_in_selected]
        
        return final_output, router_logits, router_probs

    def forward(self, x: torch.Tensor, modal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, embed_dim]
            modal_context: 可选的跨模态上下文，形状为 [batch_size, seq_len, embed_dim]
            
        Returns:
            包含layer_output, router_logits和router_probs的字典
        """
        # 传递current_labels属性（如果存在）
        if hasattr(self, 'parent_model') and self.parent_model is not None:
            if hasattr(self.parent_model, 'current_labels'):
                self.current_labels = self.parent_model.current_labels
            
        # 使用对应的实现
        if self.use_checkpoint and self.training:
            return checkpoint.checkpoint(
                self._forward_impl, 
                x, modal_context
            )
        else:
            return self._forward_impl(x, modal_context)
    
    def _forward_impl(self, x: torch.Tensor, modal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """实际的前向传播实现
        
        Args:
            x: 输入特征张量 [batch_size, seq_len, embed_dim]
            modal_context: 可选的模态上下文特征 [batch_size, context_len, embed_dim]
            
        Returns:
            包含层输出和路由信息的字典
        """
        # 保存输入以用于残差连接
        residual = x
        
        # 应用Layer Norm
        x = self.norm1(x)
        
        # 1. 注意力块
        attn_output, attention_weights = self._attention_block(x)
        
        # 应用残差连接
        x = residual + self.dropout(attn_output)
        
        # 保存新的残差
        residual = x
        
        # 2. 跨模态融合 - 应用Layer Norm
        x = self.norm2(x)
        
        # 应用跨模态融合
        fusion_output = self._cross_modal_fusion(x, modal_context)
        
        # 应用残差连接
        x = residual + self.dropout(fusion_output)
        
        # 保存新的残差
        residual = x
        
        # 3. 路由块 - 应用Layer Norm
        x = self.norm3(x)
        
        # 应用路由
        moe_output, router_logits, router_probs = self._router_block(x)
        
        # 应用残差连接
        x = residual + self.dropout(moe_output)
        
        # 创建返回字典
        output_data = {
            "layer_output": x,
            "router_logits": router_logits,
            "router_probs": router_probs,
            "expert_mask": router_probs  # 为简单起见，直接使用router_probs
        }
        
        # 可选地添加注意力权重
        if attention_weights is not None:
            output_data["attention_weights"] = attention_weights
            
        return output_data

class MultiModalMoE(nn.Module):
    """多模态混合专家模型"""
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 512,
        num_shared_experts: int = 4,
        num_modality_specific_experts: int = 2,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        num_layers: int = 6,
        activation: str = 'gelu',
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        vocab_size: int = 1000,
        max_text_len: int = 32,
        text_embed_dim: int = 128
    ):
        """初始化多模态MoE模型
        
        Args:
            img_size: 输入图像尺寸
            patch_size: Patch大小
            in_channels: 输入图像通道数
            num_classes: 类别数
            embed_dim: 嵌入维度
            num_shared_experts: 共享专家数量
            num_modality_specific_experts: 模态特定专家数量
            top_k: 每个token选择的专家数量
            dropout: Dropout比率
            num_heads: 注意力头数量
            num_layers: 层数
            activation: 激活函数
            layer_norm_eps: Layer Norm的epsilon
            initializer_range: 初始化范围
            use_gradient_checkpointing: 是否使用梯度检查点
            vocab_size: 词汇表大小
            max_text_len: 最大文本长度
            text_embed_dim: 文本嵌入维度
        """
        super().__init__()
        
        # 保存配置
        self.config = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'embed_dim': embed_dim,
            'num_shared_experts': num_shared_experts,
            'num_modality_specific_experts': num_modality_specific_experts,
            'top_k': top_k,
            'dropout': dropout,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'activation': activation,
            'layer_norm_eps': layer_norm_eps,
            'initializer_range': initializer_range,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'vocab_size': vocab_size,
            'max_text_len': max_text_len,
            'text_embed_dim': text_embed_dim
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
        
        # 位置嵌入 - 使用patch_embed计算的序列长度而不是手动计算
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        
        # 添加模态类型嵌入
        self.token_type_embed = nn.Embedding(2, embed_dim)  # 0表示视觉token，1表示文本token
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 是否使用CLS token
        self.use_cls_token = False
        
        # 文本嵌入层（简化版）
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim)
        
        # 文本位置嵌入 - 使用max(max_text_len, 77)确保可以处理较长的文本
        max_pos_len = max(max_text_len, 77)  # CLIP默认使用77，确保能处理较长的序列
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_pos_len, embed_dim))
        
        # 创建编码器层
        self.layers = nn.ModuleList([
            UnifiedModalEncoder(
                embed_dim=embed_dim,
                num_shared_experts=num_shared_experts,
                num_modality_specific_experts=num_modality_specific_experts,
                top_k=top_k,
                dropout=dropout,
                num_heads=num_heads,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                use_checkpoint=use_gradient_checkpointing
            )
            for _ in range(num_layers)
        ])
        
        # 设置编码器层的parent_model属性，以便它们能访问当前模型的属性
        for layer in self.layers:
            layer.parent_model = self
        
        # 最终的Layer Norm
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 路由损失系数
        self.router_z_loss = 0.001  # 抑制激活量
        self.router_aux_loss = 0.01  # 负载平衡损失
        
        # 初始化
        self.apply(self._init_weights)
        
        # 正则化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=initializer_range)
        nn.init.trunc_normal_(self.text_pos_embed, std=initializer_range)
        
        # 初始化当前批次的标签
        self.current_labels = None
        self.debug_forward = False  # 前向传播调试标志
        
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
        """对位置编码进行插值，使其适应不同大小的输入

        Args:
            x: 输入特征，形状为 [batch_size, num_patches, embed_dim]
            pos_embed: 位置编码，形状为 [1, orig_num_patches, embed_dim]

        Returns:
            插值后的位置编码
        """
        # 获取当前序列长度（patch数量）和维度
        _, seq_len, dim = x.shape
        # 原始位置编码的序列长度
        _, orig_seq_len, _ = pos_embed.shape
        
        # 如果序列长度相同，直接返回原始位置编码
        if seq_len == orig_seq_len:
            return pos_embed
            
        # 计算原始图像的patch数量的平方根（假设是正方形图像）
        orig_size = int(math.sqrt(orig_seq_len))
        # 计算新图像的patch数量的平方根
        new_size = int(math.sqrt(seq_len))
        
        if orig_size == new_size:
            return pos_embed
            
        # 调整位置编码以匹配新的序列长度
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, 
            size=(new_size, new_size), 
            mode='bicubic', 
            align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)
        return pos_embed
        
    def forward(self, x: torch.Tensor, text_tokens=None, attention_mask=None, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        # 获取设备信息
        device = x.device
        batch_size = x.shape[0]
        
        # 确保所有组件都在同一个设备上
        if not hasattr(self, '_device_checked'):
            self._device_checked = True
            self.to(device)
            print(f"模型已移动到设备: {device}")
        
        # 保存当前batch的标签，如果是训练模式并且在之前设置过标签
        if hasattr(self, 'current_labels') and self.current_labels is not None:
            # 确保标签在正确的设备上
            if self.current_labels.device != device:
                self.current_labels = self.current_labels.to(device)
            
            # 确保标签的批次大小与当前图像批次大小一致
            if len(self.current_labels) != batch_size:
                print(f"警告: 当前标签的批次大小({len(self.current_labels)})与图像批次大小({batch_size})不一致")
                if len(self.current_labels) > batch_size:
                    self.current_labels = self.current_labels[:batch_size]
                else:
                    padding = torch.zeros(batch_size - len(self.current_labels), 
                                       dtype=self.current_labels.dtype, 
                                       device=device)
                    self.current_labels = torch.cat([self.current_labels, padding])
            
            # 确保标签值在有效范围内
            if self.current_labels.max() >= self.config['num_classes']:
                print(f"警告: 标签值超出有效范围 [0-{self.config['num_classes']-1}]")
                self.current_labels = torch.clamp(self.current_labels, 0, self.config['num_classes']-1)
        
        # 记录初始图像尺寸
        if hasattr(self, 'debug_forward') and self.debug_forward:
            print(f"输入图像形状: {x.shape}")
            if hasattr(self, 'current_labels') and self.current_labels is not None:
                print(f"当前标签形状: {self.current_labels.shape}, 设备: {self.current_labels.device}")
                print(f"标签值范围: [{self.current_labels.min().item()}-{self.current_labels.max().item()}]")
        
        # 图像嵌入
        x = self.patch_embed(x)
        
        # 应用位置编码
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        else:
            pos_embed = self.pos_embed
        
        # 确保位置编码在正确的设备上
        pos_embed = pos_embed.to(device)
        
        # 加上位置编码
        x = x + pos_embed
        
        # 添加类型编码（图像token为0）
        token_type_ids = torch.zeros(x.size(0), x.size(1), dtype=torch.long, device=device)
        x = x + self.token_type_embed(token_type_ids)
        
        # 应用dropout
        x = self.dropout(x)
        
        # 处理文本输入
        modal_context = None
        if text_tokens is not None:
            try:
                # 确保文本数据在正确的设备上
                text_tokens = text_tokens.to(device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                else:
                    attention_mask = torch.ones_like(text_tokens, device=device)
                
                # 文本嵌入
                text_embeds = self.text_embedding(text_tokens)
                text_features = self.text_projection(text_embeds)
                
                # 检查文本特征的批次大小
                text_batch_size = text_features.shape[0]
                
                if text_batch_size != batch_size:
                    if (hasattr(self, 'current_labels') and self.current_labels is not None 
                            and text_batch_size == self.config['num_classes']):
                        # 使用当前标签选择对应的文本特征
                        valid_labels = torch.clamp(self.current_labels, 0, text_batch_size-1).long()
                        text_features = text_features[valid_labels]
                        attention_mask = attention_mask[valid_labels]
                    else:
                        raise ValueError(f"文本特征批次大小({text_batch_size})与图像批次大小({batch_size})不匹配")
                
                # 应用文本位置编码
                text_seq_len = text_features.size(1)
                if text_seq_len != self.text_pos_embed.size(1):
                    if text_seq_len < self.text_pos_embed.size(1):
                        text_pos_embed = self.text_pos_embed[:, :text_seq_len, :]
                    else:
                        last_pos = self.text_pos_embed[:, -1:, :]
                        extra_pos = last_pos.repeat(1, text_seq_len - self.text_pos_embed.size(1), 1)
                        text_pos_embed = torch.cat([self.text_pos_embed, extra_pos], dim=1)
                else:
                    text_pos_embed = self.text_pos_embed
                
                # 确保文本位置编码在正确的设备上
                text_pos_embed = text_pos_embed.to(device)
                text_features = text_features + text_pos_embed[:, :text_features.size(1), :]
                
                # 添加文本类型编码（文本token为1）
                text_type_ids = torch.ones(text_features.size(0), text_features.size(1), 
                                         dtype=torch.long, device=device)
                text_features = text_features + self.token_type_embed(text_type_ids)
                
                # 应用dropout
                text_features = self.dropout(text_features)
                modal_context = text_features
                
            except Exception as e:
                print(f"处理文本特征时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                modal_context = None
        
        # 初始化融合层输出字典
        fusion_outputs = {}
        
        # 遍历编码器层
        for layer_idx, layer in enumerate(self.layers):
            # 确保层知道当前的标签
            if hasattr(self, 'current_labels'):
                layer.current_labels = self.current_labels
            
            # 应用编码器层
            layer_outputs = layer(x, modal_context)
            x = layer_outputs['layer_output']
            
            # 存储该层的输出
            fusion_outputs[f'layer_{layer_idx}'] = {
                'router_logits': layer_outputs.get('router_logits', None),
                'router_probs': layer_outputs.get('router_probs', None),
                'expert_mask': layer_outputs.get('expert_mask', None),
                'layer_output': layer_outputs['layer_output']
            }
            
            if return_attention and 'attention_weights' in layer_outputs:
                fusion_outputs[f'layer_{layer_idx}']['attention_weights'] = layer_outputs['attention_weights']
        
        # 应用最终的Layer Norm
        x = self.norm(x)
        
        # 取[CLS] token或平均池化
        if self.use_cls_token:
            x = x[:, 0]
        else:
            x = x.mean(dim=1)
        
        # 应用分类头
        logits = self.classifier(x)
        
        # 计算路由损失
        router_loss = torch.tensor(0.0, device=device)
        for layer_idx in range(len(self.layers)):
            layer_outputs = fusion_outputs[f'layer_{layer_idx}']
            if 'router_logits' in layer_outputs and 'router_probs' in layer_outputs:
                router_loss = router_loss + self.router_z_loss * self.compute_z_loss(layer_outputs['router_logits'])
                router_loss = router_loss + self.router_aux_loss * self.compute_load_loss(layer_outputs['router_probs'])
        
        # 构建返回字典
        outputs = {
            'logits': logits,
            'embeddings': x,
            'router_loss': router_loss
        }
        
        if return_attention:
            outputs['fusion_outputs'] = fusion_outputs
        
        return outputs

    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算z损失来正则化路由逻辑

        Args:
            router_logits: 路由逻辑，形状为 [batch_size, seq_len, num_experts]

        Returns:
            z损失值
        """
        # 计算每个专家的平均路由概率
        router_probs = torch.softmax(router_logits, dim=-1)
        # 计算router_z的平方（用于正则化）
        mean_probs = router_probs.mean(dim=(0, 1))
        router_z = torch.mean(mean_probs ** 2) * router_probs.shape[-1]
        return router_z

    def compute_load_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """计算负载平衡损失，以确保专家的使用均衡

        Args:
            router_probs: 路由概率，形状为 [batch_size, seq_len, num_experts]

        Returns:
            负载平衡损失值
        """
        # 获取批量大小、序列长度和专家数量
        batch_size, seq_len, num_experts = router_probs.shape
        
        # 计算每个专家的使用频率
        # 对于每个token，计算其路由到每个专家的概率
        expert_usage = router_probs.mean(dim=(0, 1))
        
        # 计算理想的均匀分布（每个专家应该接收相同比例的token）
        ideal_usage = torch.ones_like(expert_usage) / num_experts
        
        # 使用KL散度计算与理想分布的差异
        load_loss = torch.sum(expert_usage * torch.log(expert_usage / ideal_usage))
        return load_loss

    def set_labels(self, labels: torch.Tensor):
        """设置当前批次的标签，用于文本特征的匹配
        
        参数:
            labels: 形状为 [batch_size] 的张量，包含当前批次每个样本的类别标签
        """
        self.current_labels = labels
        return self
    
    def enable_debug(self, enable: bool = True):
        """启用或禁用调试模式
        
        参数:
            enable: 是否启用调试模式
        """
        self.debug_forward = enable
        # 同时为所有编码器层设置调试标志
        for layer in self.layers:
            if hasattr(layer, 'debug_cross_modal'):
                layer.debug_cross_modal = enable
        return self

class ModelWrapper(nn.Module):
    """统一模型接口"""
    def __init__(self, model: nn.Module, preprocess: transforms.Compose):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        
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