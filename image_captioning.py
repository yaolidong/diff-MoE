import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from PIL import Image
import math

from model import (
    ImageEncoder, TextEncoder, CrossModalFusion, PatchEmbed, 
    ModelConfig, ModelWrapper, Expert
)

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 跨注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed Forward网络
        self.feed_forward = Expert(
            n_embd=d_model,
            expansion_factor=4,
            dropout=dropout,
            activation=activation
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 自注意力
        tgt_norm = self.norm1(tgt)
        tgt2, self_attn_weights = self.self_attn(
            query=tgt_norm,
            key=tgt_norm,
            value=tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # 跨注意力
        tgt_norm = self.norm2(tgt)
        tgt2, cross_attn_weights = self.cross_attn(
            query=tgt_norm,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        
        # Feed Forward
        tgt_norm = self.norm3(tgt)
        tgt2 = self.feed_forward(tgt_norm)
        tgt = tgt + self.dropout(tgt2)
        
        return tgt, {
            'self_attn_weights': self_attn_weights,
            'cross_attn_weights': cross_attn_weights
        }

class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        # 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps
            )
            for _ in range(num_layers)
        ])
        
        # 最终Layer Norm
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """生成上三角掩码避免关注未来token"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        # 如果未提供tgt_mask，生成自回归掩码
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        output = tgt
        all_attn_weights = {}
        
        # 通过各个解码器层
        for i, layer in enumerate(self.layers):
            output, attn_weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            all_attn_weights[f'layer_{i}'] = attn_weights
        
        # 最终Layer Norm
        output = self.norm(output)
        
        return output, {'attn_weights': all_attn_weights}

class ImageCaptioningMoE(nn.Module):
    """多模态图像摘要生成MoE模型 - 适用于Flickr8k等数据集"""
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int = 512,
        vocab_size: int = 10000,
        max_seq_len: int = 77,
        num_general_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        num_heads: int = 8,
        img_encoder_layers: int = 6,
        text_encoder_layers: int = 4,
        fusion_layers: int = 3,
        decoder_layers: int = 6,
        use_checkpoint: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        """初始化多模态图像摘要生成MoE模型
        
        Args:
            img_size: 输入图像大小
            patch_size: 分块大小
            in_channels: 输入图像通道数
            embed_dim: 嵌入维度
            vocab_size: 词汇表大小
            max_seq_len: 最大序列长度
            num_general_experts: 一般专家数量
            top_k: 路由选择的专家数量
            dropout: Dropout率
            num_heads: 注意力头数量
            img_encoder_layers: 图像编码器层数
            text_encoder_layers: 文本编码器层数
            fusion_layers: 融合层数
            decoder_layers: 解码器层数
            use_checkpoint: 是否使用梯度检查点
            pad_token_id: 填充token ID
            bos_token_id: 序列开始token ID
            eos_token_id: 序列结束token ID
        """
        super().__init__()
        
        # 保存配置
        self.config = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'embed_dim': embed_dim,
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'num_general_experts': num_general_experts,
            'top_k': top_k,
            'dropout': dropout,
            'num_heads': num_heads,
            'img_encoder_layers': img_encoder_layers,
            'text_encoder_layers': text_encoder_layers,
            'fusion_layers': fusion_layers,
            'decoder_layers': decoder_layers,
            'use_checkpoint': use_checkpoint,
            'pad_token_id': pad_token_id,
            'bos_token_id': bos_token_id,
            'eos_token_id': eos_token_id,
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
        self.img_seq_length = self.patch_embed.num_patches
        
        # 图像位置嵌入
        self.img_pos_embed = nn.Parameter(torch.zeros(1, self.img_seq_length, embed_dim))
        
        # 文本嵌入层
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 文本位置嵌入
        self.text_pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
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
        
        # 文本生成解码器
        self.decoder = TransformerDecoder(
            num_layers=decoder_layers,
            d_model=embed_dim,
            num_heads=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=1e-5
        )
        
        # 输出映射层
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # 初始化
        self.apply(self._init_weights)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.img_pos_embed, std=self.config['initializer_range'])
        nn.init.trunc_normal_(self.text_pos_embed, std=self.config['initializer_range'])
        
        # 特殊token IDs
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # 调整损失权重
        self.router_z_loss_weight = 0.0001  # 降低路由器正则化损失权重
        self.router_balance_loss_weight = 0.001  # 降低路由器平衡损失权重
        
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
            
    def encode_image(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码图像"""
        batch_size = images.shape[0]
        device = images.device
        
        # 图像Patch嵌入
        img_tokens = self.patch_embed(images)
        
        # 应用位置编码
        img_tokens = img_tokens + self.img_pos_embed
        
        # 应用dropout
        img_tokens = self.dropout(img_tokens)
        
        # 图像编码器
        img_encoder_outputs = self.image_encoder(img_tokens)
        img_features = img_encoder_outputs['output']
        
        return {
            'features': img_features,
            'encoder_outputs': img_encoder_outputs
        }
    
    def encode_text(self, text_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码文本"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # 文本嵌入
        text_features = self.text_embedding(text_tokens)
        
        # 应用位置编码
        max_len = min(text_features.size(1), self.text_pos_embed.size(1))
        text_features[:, :max_len] = text_features[:, :max_len] + self.text_pos_embed[:, :max_len]
        
        # 应用dropout
        text_features = self.dropout(text_features)
        
        # 文本编码器
        text_encoder_outputs = self.text_encoder(text_features)
        text_features = text_encoder_outputs['output']
        
        return {
            'features': text_features,
            'encoder_outputs': text_encoder_outputs
        }
    
    def fuse_features(self, img_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """融合图像和文本特征"""
        # 跨模态融合
        fusion_outputs = self.cross_modal_fusion(img_features, text_features)
        fused_features = fusion_outputs['output']
        
        return {
            'features': fused_features,
            'fusion_outputs': fusion_outputs
        }
        
    def forward(self, images: torch.Tensor, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """前向传播
        
        Args:
            images: [batch_size, in_channels, height, width]
            text_tokens: [batch_size, seq_len] - 输入文本(已填充)
            attention_mask: [batch_size, seq_len] - 文本注意力掩码
            
        Returns:
            包含预测和损失的字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        try:
            # 编码图像
            img_output = self.encode_image(images)
            img_features = img_output['features']
            
            # 切分输入文本和目标文本
            decoder_input = text_tokens[:, :-1]  # 从第一个token开始的文本作为输入
            decoder_target = text_tokens[:, 1:]  # 从第二个token开始的文本作为目标
            
            # 创建decoder的key padding mask
            if attention_mask is not None:
                decoder_padding_mask = attention_mask[:, :-1]
            else:
                decoder_padding_mask = None
            
            # 获取文本嵌入
            decoder_input_embeds = self.text_embedding(decoder_input)
            
            # 应用位置编码
            max_len = min(decoder_input_embeds.size(1), self.text_pos_embed.size(1))
            decoder_input_embeds[:, :max_len] = decoder_input_embeds[:, :max_len] + self.text_pos_embed[:, :max_len]
            
            # 应用dropout
            decoder_input_embeds = self.dropout(decoder_input_embeds)
            
            # 解码器前向传播
            decoder_outputs, decoder_attn_weights = self.decoder(
                tgt=decoder_input_embeds,
                memory=img_features,
                tgt_key_padding_mask=decoder_padding_mask if decoder_padding_mask is not None else None,
                memory_key_padding_mask=None  # 图像特征没有填充
            )
            
            # 映射到词汇表大小
            logits = self.output_projection(decoder_outputs)
            
            # 计算交叉熵损失
            loss = None
            if decoder_target is not None:
                # 创建损失掩码，忽略填充标记
                loss_mask = (decoder_target != self.pad_token_id).float()
                
                # 计算每个token位置的损失
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(
                    logits.reshape(-1, logits.size(-1)),
                    decoder_target.reshape(-1)
                )
                
                # 应用掩码并计算平均损失
                token_losses = token_losses.reshape(batch_size, -1) * loss_mask
                loss = token_losses.sum() / loss_mask.sum().clamp(min=1.0)
            
            # 收集所有层的路由决策
            all_router_logits = []
            all_router_probs = []
            
            # 收集图像编码器路由决策
            img_encoder_outputs = img_output['encoder_outputs']
            for i in range(self.config['img_encoder_layers']):
                layer_outputs = img_encoder_outputs['layer_outputs'][f'layer_{i}']
                if 'router_logits' in layer_outputs and layer_outputs['router_logits'] is not None:
                    all_router_logits.append(layer_outputs['router_logits'])
                if 'router_probs' in layer_outputs and layer_outputs['router_probs'] is not None:
                    all_router_probs.append(layer_outputs['router_probs'])
            
            # 计算路由损失
            router_z_loss = torch.tensor(0.0, device=device)
            router_balance_loss = torch.tensor(0.0, device=device)
            
            # 只在有路由决策时计算损失
            if all_router_logits:
                for router_logits in all_router_logits:
                    router_z_loss = router_z_loss + self.compute_z_loss(router_logits)
                router_z_loss = router_z_loss / len(all_router_logits)
            
            if all_router_probs:
                for router_probs in all_router_probs:
                    router_balance_loss = router_balance_loss + self.compute_load_loss(router_probs)
                router_balance_loss = router_balance_loss / len(all_router_probs)
            
            # 总路由损失
            router_loss = self.router_z_loss_weight * router_z_loss + self.router_balance_loss_weight * router_balance_loss
            
            # 总损失
            total_loss = loss + router_loss if loss is not None else router_loss
            
            # 检查损失值是否有效
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise ValueError("检测到无效的损失值")
            
            return {
                'logits': logits,
                'loss': loss,
                'router_z_loss': router_z_loss,
                'router_balance_loss': router_balance_loss,
                'router_loss': router_loss,
                'total_loss': total_loss,
                'decoder_attentions': decoder_attn_weights
            }
            
        except Exception as e:
            print(f"前向传播时发生错误: {str(e)}")
            raise
    
    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """计算路由器正则化损失
        
        Args:
            router_logits: [batch_size * seq_len, num_experts] 路由器logits
            
        Returns:
            z_loss: 标量损失值
        """
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size * seq_len, num_experts]
        return torch.mean(torch.sum(router_probs * (router_logits - router_logits.logsumexp(dim=-1, keepdim=True)), dim=-1))
        
    def compute_load_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """计算负载平衡损失，以确保专家的使用均衡
        
        Args:
            router_probs: [batch_size * seq_len, num_experts] 路由概率
            
        Returns:
            load_loss: 标量损失值
        """
        # 计算每个专家的负载
        num_tokens = router_probs.shape[0]  # batch_size * seq_len
        expert_load = router_probs.sum(0)  # [num_experts]
        
        # 计算每个专家的期望负载
        expected_load = torch.ones_like(expert_load) * (num_tokens / expert_load.size(0))
        
        # 计算负载差异
        load_diff = torch.abs(expert_load - expected_load)
        
        # 计算负载平衡损失
        load_loss = torch.mean(load_diff)
        
        return load_loss
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 30,
        min_length: int = 5,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        early_stopping: bool = True
    ) -> Dict[str, Any]:
        """生成图像描述
        
        Args:
            images: [batch_size, in_channels, height, width]
            max_length: 生成序列的最大长度
            min_length: 生成序列的最小长度
            num_beams: 束搜索的束数
            temperature: 采样温度
            top_k: top-k采样参数
            top_p: nucleus采样参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            no_repeat_ngram_size: 不重复n-gram的大小
            early_stopping: 是否提前停止
            
        Returns:
            包含生成结果的字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 编码图像
        img_output = self.encode_image(images)
        img_features = img_output['features']
        
        # 初始化解码序列
        input_ids = torch.full(
            (batch_size, 1),
            self.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 生成文本
        for i in range(max_length):
            # 获取文本嵌入
            decoder_input_embeds = self.text_embedding(input_ids)
            
            # 应用位置编码
            max_len = min(decoder_input_embeds.size(1), self.text_pos_embed.size(1))
            decoder_input_embeds[:, :max_len] = decoder_input_embeds[:, :max_len] + self.text_pos_embed[:, :max_len]
            
            # 应用dropout
            decoder_input_embeds = self.dropout(decoder_input_embeds)
            
            # 解码器前向传播
            decoder_outputs, _ = self.decoder(
                tgt=decoder_input_embeds,
                memory=img_features
            )
            
            # 获取最后一个token的输出
            last_decoder_output = decoder_outputs[:, -1]
            
            # 映射到词汇表大小
            next_token_logits = self.output_projection(last_decoder_output)
            
            # 应用温度
            next_token_logits = next_token_logits / temperature
            
            # 重复惩罚
            if repetition_penalty > 1.0:
                for batch_idx in range(batch_size):
                    for token_idx in input_ids[batch_idx]:
                        if token_idx != self.pad_token_id:
                            next_token_logits[batch_idx, token_idx] /= repetition_penalty
            
            # 应用Top-K筛选
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 应用Top-p筛选
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累积超过top_p的标记
                sorted_indices_to_remove = cumulative_probs > top_p
                # 将第一个标记保留（不移除）
                sorted_indices_to_remove[..., 0] = False
                
                # 为每个batch样本创建一个索引掩码
                for batch_idx in range(batch_size):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 最小长度约束
            if i < min_length:
                next_token[next_token == self.eos_token_id] = -100  # 暂时将EOS替换为无效ID
            
            # 将新token添加到序列中
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否所有序列都生成了EOS
            if (next_token == self.eos_token_id).all() and i >= min_length:
                break
        
        # 删除BOS token并返回生成的序列
        generated_ids = input_ids[:, 1:]  # 去除BOS token
        
        return {
            'generated_ids': generated_ids,
            'img_features': img_features
        }

# 创建Flickr8k模型
def create_flickr8k_model(pretrained=False):
    """创建适用于Flickr8k数据集的MoE模型"""
    model = ImageCaptioningMoE(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        vocab_size=10000,  # Flickr8k词汇表大小
        max_seq_len=77,
        num_general_experts=8,
        top_k=2,
        dropout=0.1,
        num_heads=8,
        img_encoder_layers=6,
        text_encoder_layers=4,
        fusion_layers=3,
        decoder_layers=6,
        use_checkpoint=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2
    )
    
    # 预处理变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )  # ImageNet标准化
    ])
    
    # 如果是预训练模型，加载权重
    if pretrained:
        # 加载预训练权重的代码
        pass
    
    return ModelWrapper(model, transform)

# 示例用法
if __name__ == "__main__":
    # 创建Flickr8k模型
    flickr_model = create_flickr8k_model()
    
    # 使用示例
    dummy_img = torch.randn(2, 3, 224, 224)
    dummy_text = torch.randint(0, 1000, (2, 20))  # [batch_size, seq_len]
    
    # 前向传播
    with torch.no_grad():
        outputs = flickr_model.model(dummy_img, dummy_text)
    
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # 生成示例
    with torch.no_grad():
        gen_outputs = flickr_model.model.generate(dummy_img)
    
    print(f"Generated ids shape: {gen_outputs['generated_ids'].shape}") 