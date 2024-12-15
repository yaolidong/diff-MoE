import torch
import torch.nn as nn
from moe_encoders import ImageMoE, TextMoE 
from cross_attention import CrossAttention
from TextDecoder import TextDecoder

class DualTowerModel(nn.Module):
    def __init__(self, vocab_size, output_dim=1024, n_head=8, num_classes=10, max_text_length=16):
        super().__init__()
        # 增加图像backbone的复杂度
        self.image_tower = ImageMoE(
            img_size=32,
            patch_size=4,
            in_channels=3,
            embed_dim=1024,
            num_experts=16,  # 增加专家数量
            top_k=4  # 增加每个token使用的专家数
        )
        
        # 增加文本编码器的复杂度
        self.text_tower = TextMoE(
            vocab_size,
            seq_length=16,
            embed_dim=1024,
            num_experts=16,
            top_k=4
        )
        
        # 添加更复杂的跨模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 添加跨模态注意力层
        self.img2text_attention = CrossAttention(output_dim, n_head)
        self.text2img_attention = CrossAttention(output_dim, n_head)

        # 添加文本解码器
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            max_length=max_text_length,
            embed_dim=output_dim
        )

        # 修改分类头，确保输出维度正确
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, num_classes)  # 移除 LogSoftmax
        )
        
    def forward(self, images, input_ids=None, attention_mask=None):
        # 添加默认值处理
        if input_ids is None:
            batch_size = images.size(0)
            input_ids = torch.zeros((batch_size, 16), dtype=torch.long, device=images.device)
            attention_mask = torch.ones((batch_size, 16), dtype=torch.long, device=images.device)
            
        # 获取基础特征
        first_image_output, second_image_output, image_feature_vector, image_cls, image_expert_outputs, image_gating_outputs = self.image_tower(images)
        first_text_output, second_text_output, text_feature_vector, text_cls = self.text_tower(input_ids, attention_mask)
        
        # 跨模态注意力
        img2text_features = self.img2text_attention(text_feature_vector, image_feature_vector)
        text2img_features = self.text2img_attention(image_feature_vector, text_feature_vector)
        
        # 特征融合
        fused_features = self.fusion_layer(torch.cat([img2text_features, text2img_features], dim=-1))
        
        # 解码文本
        text_reconstruction = self.text_decoder(fused_features)
        
        # 分类预测
        fused_features = self.fusion_layer(torch.cat([img2text_features, text2img_features], dim=-1))
        fused_cls = self.classifier(fused_features)  # [batch_size, num_classes]
        
        # 确保输出维度正确
        if len(fused_cls.shape) == 3:
            fused_cls = fused_cls.squeeze(1)  # 移除多余的维度
            
        # 打印维度信息
        print(f"fused_features shape: {fused_features.shape}")
        print(f"fused_cls shape: {fused_cls.shape}")
        
        return (image_feature_vector, text_feature_vector, 
                image_cls, text_cls, fused_cls,
                text_reconstruction,
                (image_expert_outputs, None), (image_gating_outputs, None)) 