import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass
from AttentiveRouter import AttentiveRouter
from export import Expert
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.checkpoint import checkpoint

# 配置日志
logger = logging.getLogger(__name__)

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
    use_gradient_checkpointing: bool = True

class PatchEmbed(nn.Module):
    """图像转Patch嵌入"""
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入图像大小应为 {self.img_size}x{self.img_size}, 但得到 {H}x{W}"
        
        x = self.proj(x)  # [B, embed_dim, grid_size, grid_size]
        return x

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
        super().__init__()
        
        # 计算每个注意力头的维度
        assert embed_dim % num_heads == 0, "嵌入维度必须能被注意力头数整除"
        self.head_size = embed_dim // num_heads
        
        # 添加跨模态融合层
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            batch_first=True
        )
        self.cross_modal_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 专家层
        self.shared_experts = nn.ModuleList([
            Expert(embed_dim, 
                  dropout=dropout, 
                  activation=activation) 
            for _ in range(num_shared_experts)
        ])
        self.image_specific_experts = nn.ModuleList([
            Expert(embed_dim, 
                  dropout=dropout, 
                  activation=activation)
            for _ in range(num_modality_specific_experts)
        ])
        
        # 路由器 - 使用改进的路由策略
        self.shared_router = AttentiveRouter(embed_dim, top_k, num_shared_experts)
        self.image_router = AttentiveRouter(embed_dim, top_k, num_modality_specific_experts)
        
        # 使用PyTorch内置的多头注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            batch_first=True
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络 - 使用更大的扩展倍数
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=use_bias),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim, bias=use_bias),
            nn.Dropout(dropout)
        )
        
        # 添加模态融合门控机制
        self.modal_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # 保存配置
        self.config = {
            'embed_dim': embed_dim,
            'num_shared_experts': num_shared_experts,
            'num_modality_specific_experts': num_modality_specific_experts,
            'top_k': top_k,
            'dropout': dropout,
            'num_heads': num_heads,
            'activation': activation,
            'layer_norm_eps': layer_norm_eps,
            'use_bias': use_bias,
            'initializer_range': 0.02
        }
        
        # 梯度检查点标志
        self.use_checkpoint = use_checkpoint
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=self.config['initializer_range'])
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _attention_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """注意力块，用于梯度检查点"""
        residual = x
        x = self.norm1(x)
        attn_output, attn_weights = self.self_attention(x, x, x)
        x = residual + self.dropout(attn_output)
        return x, attn_weights
    
    def _cross_modal_fusion(self, x: torch.Tensor, modal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """跨模态融合层，用于整合不同模态的信息"""
        if modal_context is None:
            return x
            
        residual = x
        x = self.cross_modal_norm(x)
        
        # 添加形状检查和调试信息
        batch_size, seq_len, embed_dim = x.shape
        modal_batch, modal_seq, modal_dim = modal_context.shape
        
        # 确保维度匹配
        if embed_dim != modal_dim:
            raise ValueError(f"嵌入维度不匹配: x={embed_dim}, modal_context={modal_dim}")
        
        # 处理批次大小不匹配的情况
        # 如果modal_context的批次大小与x不同，则需要调整
        if modal_batch != batch_size:
            # 如果modal_context是类别描述（即批次大小为类别数量）
            # 则为每个样本复制相同的modal_context
            if modal_batch == 10:  # 通常是10个类别
                # 创建一个新的modal_context，批次大小与x相同
                new_modal_context = torch.zeros(batch_size, modal_seq, modal_dim, 
                                             device=modal_context.device, 
                                             dtype=modal_context.dtype)
                # 为每个样本复制相同的modal_context
                new_modal_context = modal_context[0:1].expand(batch_size, modal_seq, modal_dim)
                modal_context = new_modal_context
                modal_batch = batch_size
        
        # 使用注意力机制进行跨模态融合
        try:
            # 如果modal_context只有一个序列元素，尝试扩展它以匹配x的序列长度
            if modal_seq == 1:
                modal_context_expanded = modal_context.expand(modal_batch, seq_len, modal_dim)
                fusion_output, _ = self.cross_modal_attention(x, modal_context_expanded, modal_context_expanded)
            else:
                fusion_output, _ = self.cross_modal_attention(x, modal_context, modal_context)
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
        
        return fusion_output
    
    def _router_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """路由块，用于梯度检查点"""
        residual = x
        x = self.norm2(x)
        
        # 共享专家路由
        shared_router_output = self.shared_router(x)
        shared_masks = shared_router_output['masks']
        shared_loss = shared_router_output['loss']
        
        # 图像特定专家路由
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
        x = x + self.ffn(self.norm3(x))  # 使用额外的LayerNorm
        
        # 计算总路由损失
        total_router_loss = shared_loss + image_loss
        
        return x, total_router_loss, shared_router_output['expert_usage']

    def forward(self, x: torch.Tensor, modal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
            modal_context: 可选的模态上下文张量，用于多模态融合
            
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典:
                - output: 输出张量
                - router_loss: 路由损失
                - attention_weights: 注意力权重
        """        
        if self.use_checkpoint and self.training:
            # 使用梯度检查点进行注意力计算
            x, attn_weights = checkpoint(self._attention_block, x, use_reentrant=False)
            
            # 使用跨模态融合层增强特征表示
            if modal_context is not None:
                x = self._cross_modal_fusion(x, modal_context)
            
            # 使用梯度检查点进行路由计算
            x, router_loss, expert_usage = checkpoint(self._router_block, x, use_reentrant=False)
        else:
            # 自注意力
            x, attn_weights = self._attention_block(x)
            
            # 使用跨模态融合层增强特征表示
            if modal_context is not None:
                x = self._cross_modal_fusion(x, modal_context)
            
            # 专家路由
            x, router_loss, expert_usage = self._router_block(x)
        
        return {
            'output': x,
            'router_loss': router_loss,
            'attention_weights': attn_weights,
            'expert_usage': expert_usage
        }

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
        use_gradient_checkpointing: bool = True,
        vocab_size: int = 1000,
        max_text_len: int = 32,
        text_embed_dim: int = 128
    ):
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
        
        # 位置嵌入
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # 文本嵌入层（简化版）
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim)
        
        # 编码器层
        self.encoders = nn.ModuleList([
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
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self.apply(self._init_weights)
        self.pos_embed.data.normal_(mean=0.0, std=initializer_range)
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, x: torch.Tensor, text_tokens=None, return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入图像张量 [B, C, H, W]
            text_tokens: 可选的文本标记 [B, L] 或 [num_classes, L]
            return_attention: 是否返回注意力权重
            
        Returns:
            Dict: 输出字典
        """
        # Patch嵌入
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 处理文本模态输入（如果有）
        batch_size = x.shape[0]
        num_classes = self.config['num_classes']
        
        if text_tokens is not None:
            # 检查text_tokens的形状，判断是否是类别描述
            text_token_shape = text_tokens.shape
            
            # 如果是类别描述 [num_classes, seq_len]
            if text_token_shape[0] == num_classes:
                # 使用文本嵌入层处理文本
                text_embeds = self.text_embedding(text_tokens)
                # 将文本嵌入投影到与图像相同的维度空间
                # 对每个类别，将所有token取平均得到一个表示
                modal_context = self.text_projection(text_embeds.mean(dim=1, keepdim=True))
                # 注意：这里modal_context的形状是 [num_classes, 1, embed_dim]
                # 在_cross_modal_fusion中会处理批次大小不匹配的问题
            else:
                # 假设是按批次的文本数据 [batch_size, seq_len]
                text_embeds = self.text_embedding(text_tokens)
                # 将文本嵌入投影到与图像相同的维度空间
                modal_context = self.text_projection(text_embeds.mean(dim=1, keepdim=True))
        else:
            # 创建一个默认的上下文向量
            modal_context = torch.zeros(batch_size, 1, self.config['embed_dim'], 
                                      device=x.device)
        
        # 通过编码器层
        total_router_loss = 0
        attention_weights = []
        router_outputs = []
        
        for encoder in self.encoders:
            # 将两种模态都传递给编码器
            outputs = encoder(x, modal_context)
            x = outputs['output']
            total_router_loss += outputs['router_loss']
            
            if return_attention and 'attention_weights' in outputs:
                attention_weights.append(outputs['attention_weights'])
            
            if 'expert_usage' in outputs:
                router_outputs.append(outputs['expert_usage'])
        
        # 全局平均池化
        x = x.mean(dim=1)  # [B, embed_dim]
        
        # 分类
        x = self.norm(x)
        logits = self.head(x)
        
        # 构建输出字典
        outputs = {
            'logits': logits,
            'router_loss': total_router_loss
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            outputs['router_outputs'] = router_outputs
        
        return outputs

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