import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, Any, Optional, Union, List
import numpy as np
from PIL import Image

from model import ImageEncoder, PatchEmbed, ModelConfig, ModelWrapper

class ImageClassificationMoE(nn.Module):
    """单模态图像分类MoE模型 - 适用于Fashion_MNIST和CIFAR-10等数据集"""
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
        use_checkpoint: bool = False,
        pool_type: str = 'cls'  # 'cls' 或 'mean'
    ):
        """初始化单模态图像分类MoE模型
        
        Args:
            img_size: 输入图像大小
            patch_size: 分块大小
            in_channels: 输入图像通道数
            num_classes: 分类类别数
            embed_dim: 嵌入维度
            num_general_experts: 一般专家数量
            top_k: 路由选择的专家数量
            dropout: Dropout率
            num_heads: 注意力头数量
            img_encoder_layers: 图像编码器层数
            use_checkpoint: 是否使用梯度检查点
            pool_type: 池化类型，'cls'或'mean'
        """
        super().__init__()
        
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
            'use_checkpoint': use_checkpoint,
            'pool_type': pool_type,
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
        
        # CLS token (对于'cls'池化类型)
        if pool_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # 图像位置嵌入
        # 如果使用cls token，需要增加一个位置
        if pool_type == 'cls':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_length, embed_dim))
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=self.config['initializer_range'])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 图像编码器 - 直接使用 model.py 中的 ImageEncoder
        self.image_encoder = ImageEncoder(
            embed_dim=embed_dim,
            num_layers=img_encoder_layers,
            num_general_experts=num_general_experts,
            top_k=top_k,
            dropout=dropout,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint
        )
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # 保存池化类型
        self.pool_type = pool_type
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            images: [batch_size, in_channels, height, width]
            
        Returns:
            包含logits和损失的字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 图像Patch嵌入
        x = self.patch_embed(images)
        
        # 2. 添加CLS token (如果使用'cls'池化)
        if self.pool_type == 'cls':
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # 3. 添加位置编码
        x = x + self.pos_embed
        
        # 4. Dropout
        x = self.dropout(x)
        
        # 5. 图像编码器
        encoder_outputs = self.image_encoder(x)
        x = encoder_outputs['output']
        
        # 6. Layer Norm
        x = self.norm(x)
        
        # 7. 池化
        if self.pool_type == 'cls':
            # 使用CLS token作为图像表示
            pooled_output = x[:, 0]
        else:
            # 使用平均池化
            pooled_output = x.mean(dim=1)
        
        # 8. 分类
        logits = self.classifier(pooled_output)
        
        # 9. 收集路由损失
        router_loss = torch.tensor(0.0, device=device)
        
        # 从编码器输出中获取路由损失
        if 'layer_outputs' in encoder_outputs:
            for layer_idx, layer_output in encoder_outputs['layer_outputs'].items():
                if 'router_logits' in layer_output:
                    router_logits = layer_output['router_logits']
                    router_loss = router_loss + self.compute_router_loss(router_logits)
        
        return {
            'logits': logits,
            'embeddings': pooled_output,
            'router_loss': router_loss,
            'expert_activations': encoder_outputs.get('layer_outputs', {})
        }
    
    def compute_router_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算路由器损失，包括负载均衡损失和辅助损失
        
        Args:
            router_logits: [batch_size * seq_len, num_experts] 路由器logits
            
        Returns:
            router_loss: 标量损失值
        """
        # 计算路由概率
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 1. 计算负载均衡损失
        # 计算每个专家的平均使用率
        num_experts = router_probs.shape[-1]
        expert_usage = router_probs.mean(dim=0)
        # 计算与理想均匀分布的KL散度
        target_usage = torch.ones_like(expert_usage) / num_experts
        load_balance_loss = F.kl_div(
            expert_usage.log(),
            target_usage,
            reduction='batchmean'
        )
        
        # 2. 计算辅助损失 (z-loss)
        # 防止路由概率过于极端
        router_z = torch.logsumexp(router_logits, dim=-1)
        z_loss = torch.mean(router_z ** 2)
        
        # 3. 组合损失
        router_loss = load_balance_loss + 0.001 * z_loss
        
        return router_loss

# 数据集特定的模型配置

def create_fashion_mnist_model(pretrained=False):
    """创建适用于Fashion MNIST数据集的MoE模型"""
    model = ImageClassificationMoE(
        img_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=256,  # 较小的嵌入维度，适合简单数据集
        num_general_experts=4,
        top_k=2,
        dropout=0.1,
        num_heads=4,
        img_encoder_layers=4,  # 较少的层数
        use_checkpoint=False,
        pool_type='mean'  # 使用平均池化
    )
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion MNIST的均值和标准差
    ])
    
    # 如果是预训练模型，加载权重
    if pretrained:
        # 加载预训练权重的代码
        pass
    
    return ModelWrapper(model, transform)

def create_cifar10_model(pretrained=False):
    """创建适用于CIFAR-10数据集的MoE模型"""
    model = ImageClassificationMoE(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,  # 减小嵌入维度
        num_general_experts=4,  # 减少专家数量
        top_k=1,  # 每个token只路由到1个专家
        dropout=0.2,  # 增加dropout
        num_heads=4,  # 减少注意力头数
        img_encoder_layers=4,  # 减少编码器层数
        use_checkpoint=False,
        pool_type='mean'  # 使用平均池化而不是cls token
    )
    
    # 预处理变换 - 添加数据增强
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    # 如果是预训练模型，加载权重
    if pretrained:
        # 加载预训练权重的代码
        pass
    
    return ModelWrapper(model, transform)

# 示例用法
if __name__ == "__main__":
    # 创建Fashion MNIST模型
    fashion_model = create_fashion_mnist_model()
    
    # 创建CIFAR-10模型
    cifar_model = create_cifar10_model()
    
    # 使用示例
    dummy_fashion_input = torch.randn(2, 1, 28, 28)
    dummy_cifar_input = torch.randn(2, 3, 32, 32)
    
    # 前向传播
    with torch.no_grad():
        fashion_outputs = fashion_model.model(dummy_fashion_input)
        cifar_outputs = cifar_model.model(dummy_cifar_input)
    
    print(f"Fashion MNIST output shape: {fashion_outputs['logits'].shape}")
    print(f"CIFAR-10 output shape: {cifar_outputs['logits'].shape}") 