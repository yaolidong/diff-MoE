import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
from AttentiveRouter import AttentiveRouter
from mHselfAttention import Expert
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置类"""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
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

class UnifiedModalEncoder(nn.Module):
    """统一的模态编码器"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 计算每个注意力头的维度
        assert config.embed_dim % config.num_heads == 0, "嵌入维度必须能被注意力头数整除"
        self.head_size = config.embed_dim // config.num_heads
        
        # 专家层
        self.shared_experts = nn.ModuleList([
            Expert(config.embed_dim, 
                  dropout=config.dropout, 
                  activation=config.activation) 
            for _ in range(config.num_shared_experts)
        ])
        self.image_specific_experts = nn.ModuleList([
            Expert(config.embed_dim, 
                  dropout=config.dropout, 
                  activation=config.activation)
            for _ in range(config.num_modality_specific_experts)
        ])
        self.text_specific_experts = nn.ModuleList([
            Expert(config.embed_dim, 
                  dropout=config.dropout, 
                  activation=config.activation)
            for _ in range(config.num_modality_specific_experts)
        ])
        
        # 路由器
        self.shared_router = AttentiveRouter(config.embed_dim, config.top_k, config.num_shared_experts)
        self.image_router = AttentiveRouter(config.embed_dim, config.top_k, config.num_modality_specific_experts)
        self.text_router = AttentiveRouter(config.embed_dim, config.top_k, config.num_modality_specific_experts)
        
        # 使用PyTorch内置的多头注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            bias=config.use_bias,
            batch_first=True
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim, bias=config.use_bias),
            nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.embed_dim, config.embed_dim, bias=config.use_bias),
            nn.Dropout(config.dropout)
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
                - 输出张量
                - 路由损失
                - 专家分配列表
                - 注意力权重
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.self_attention(x, x, x)
        x = residual + self.dropout(attn_output)
        
        # 专家路由
        residual = x
        x = self.norm2(x)
        
        # 共享专家路由
        shared_router_output = self.shared_router(x)
        shared_masks = shared_router_output['masks']
        shared_loss = shared_router_output['loss']
        
        # 模态特定专家路由
        image_router_output = self.image_router(x)
        image_masks = image_router_output['masks']
        image_loss = image_router_output['loss']
        
        # 计算专家输出
        expert_outputs = []
        
        # 共享专家处理
        for i, expert in enumerate(self.shared_experts):
            mask = shared_masks[..., i:i+1]
            expert_output = expert(x)
            expert_outputs.append(expert_output * mask)
            
        # 图像特定专家处理
        for i, expert in enumerate(self.image_specific_experts):
            mask = image_masks[..., i:i+1]
            expert_output = expert(x)
            expert_outputs.append(expert_output * mask)
            
        # 合并所有专家输出
        combined_output = sum(expert_outputs)
        
        # 残差连接和前馈网络
        x = residual + self.dropout(combined_output)
        x = x + self.ffn(self.norm2(x))
        
        # 计算总路由损失
        total_router_loss = shared_loss + image_loss
        
        # 收集专家分配信息
        expert_assignments = [shared_masks, image_masks]
        
        return x, total_router_loss, expert_assignments, attn_weights

class MultiModalMoE(nn.Module):
    """多模态混合专家模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 图像编码
        self.patch_embed = nn.Conv2d(
            config.in_channels, 
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.use_bias
        )
        self.num_patches = (config.img_size // config.patch_size) ** 2
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, config.embed_dim))
        self.pos_drop = nn.Dropout(config.dropout)
        
        # 编码层
        self.encoders = nn.ModuleList([
            UnifiedModalEncoder(config) for _ in range(config.num_layers)
        ])
        
        # 分类头
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.embed_dim, config.num_classes, bias=config.use_bias)
        
        # 初始化
        self._init_weights()
        
        # 记录模型参数数量
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"模型总参数量: {num_params:,}")
        
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.trunc_normal_(self.pos_embed, std=self.config.initializer_range)
        self.apply(self._init_layer)
        
    def _init_layer(self, module):
        """初始化层权重"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """获取所有注意力图"""
        attention_maps = {}
        for idx, encoder in enumerate(self.encoders):
            attention_maps[f'layer_{idx}'] = encoder.self_attention
        return attention_maps
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不需要权重衰减的参数名称"""
        return {'pos_embed'}
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Dict[str, Any]:
        """前向传播
        Args:
            x: 输入张量 [B, C, H, W]
            return_attention: 是否返回注意力权重
            
        Returns:
            包含以下键的字典:
            - logits: 分类logits [B, num_classes]
            - router_loss: 路由损失
            - router_outputs: 路由器输出
            - attention_weights: 注意力权重 (如果return_attention=True)
            - expert_usages: 专家使用情况
        """
        # 保存输入维度信息
        B = x.shape[0]
        
        # 1. Patch embedding: [B, C, H, W] -> [B, embed_dim, H/patch_size, W/patch_size]
        x = self.patch_embed(x)
        
        # 2. 转换维度顺序: [B, embed_dim, H/P, W/P] -> [B, H/P, W/P, embed_dim]
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # 3. 展平空间维度: [B, H/P, W/P, embed_dim] -> [B, (H/P)*(W/P), embed_dim]
        x = x.reshape(B, -1, self.config.embed_dim)
        
        # 4. 添加位置编码
        if x.size(1) != self.pos_embed.size(1):
            raise ValueError(f"位置编码维度不匹配: x.size(1)={x.size(1)}, pos_embed.size(1)={self.pos_embed.size(1)}")
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 5. 通过编码层
        total_router_loss = 0
        router_outputs = []
        attention_weights = []
        expert_usages = []
        
        for encoder in self.encoders:
            # 确保每次进入编码器前张量都是连续的
            x = x.contiguous()
            # 编码器处理
            x, router_loss, expert_usage, attn_weights = encoder(x)
            # 累积损失和输出
            total_router_loss += router_loss
            router_outputs.append(expert_usage)
            if return_attention:
                attention_weights.append(attn_weights)
            expert_usages.extend(expert_usage)
        
        # 6. 全局平均池化: [B, seq_len, embed_dim] -> [B, embed_dim]
        x = x.mean(dim=1)
        
        # 7. 最终的分类头
        x = self.norm(x.contiguous())
        logits = self.head(x)
        
        # 8. 构建输出字典
        outputs = {
            'logits': logits,
            'router_loss': total_router_loss,
            'router_outputs': router_outputs,
            'expert_usages': expert_usages
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs 