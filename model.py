import torch
import torch.nn as nn
from moe_encoders import ImageMoE, TextMoE 
from cross_attention import CrossAttention


class DualTowerModel(nn.Module):
    def __init__(self, vocab_size, output_dim=1024, n_head=8, num_classes = 10):
        super().__init__()
        self.image_tower = ImageMoE()
        self.text_tower = TextMoE(vocab_size)
        self.cross_attention = CrossAttention(output_dim, n_head)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, images, input_ids, attention_mask):
        first_image_output, second_image_output, image_feature_vector, image_cls = self.image_tower(images)
        first_text_output, second_text_output, text_feature_vector, text_cls = self.text_tower(input_ids, attention_mask)
        
        # 应用交叉注意力
        cross_attention_output = self.cross_attention(image_feature_vector, text_feature_vector)
        
        cross_attention_output = cross_attention_output.mean(dim=1)  # 平均池化
        
        # 使用 classifier 层生成分类预测
        classification_output = self.classifier(cross_attention_output)
        
        return image_feature_vector, text_feature_vector, image_cls, text_cls
