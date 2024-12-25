import torch
import torch.nn as nn
from moe_encoders import UnifiedModalEncoder

class MultiModalMoE(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 vocab_size=30522, seq_length=77, embed_dim=1024,
                 num_experts=8, top_k=2, num_heads=8, num_layers=6,
                 num_classes=10):
        super().__init__()
        
        # 统一的模态编码器
        self.unified_encoder = UnifiedModalEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            vocab_size=vocab_size,
            seq_length=seq_length,
            embed_dim=embed_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_layers=num_layers
        )
        
        # 分类头 - 直接输出类别数量的logits
        self.classification = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, images=None, input_ids=None, attention_mask=None):
        # 获取统一特征
        outputs = self.unified_encoder(image=images, input_ids=input_ids, attention_mask=attention_mask)
        
        # 分类 - 确保输出维度正确
        feature_vector = outputs['feature_vector']
        if len(feature_vector.shape) == 3:  # 如果特征向量是3D的 [batch, seq_len, embed_dim]
            feature_vector = feature_vector.mean(dim=1)  # 对序列维度取平均
        logits = self.classification(feature_vector)  # [batch_size, num_classes]
        
        # 更新输出字典
        outputs['logits'] = logits
        
        return outputs
    
    def get_expert_stats(self):
        """获取专家使用统计"""
        return self.unified_encoder.get_expert_stats()
    
    def reset_expert_stats(self):
        """重置专家使用统计"""
        self.unified_encoder.reset_expert_stats() 