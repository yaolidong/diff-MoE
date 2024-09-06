import torch
import torch.nn as nn
from moe_encoders import ImageMoE, TextMoE  # 导入新的编码器
from cross_attention import CrossAttention


class DualTowerModel(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.image_tower = ImageMoE()
        self.text_tower = TextMoE(vocab_size)
        self.cross_attention = CrossAttention(128, 8)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_first_vector, image_second_vector, image_global_vector, image_cls = self.image_tower(images)
        text_first_vector, text_second_vector, sentence_vector = self.text_tower(input_ids, attention_mask)


        # 应用交叉注意力
        cross_attention_output = self.cross_attention(image_global_vector, sentence_vector)
        cross_attention_output = cross_attention_output.mean(dim=1)  # 平均池化
        
        # 使用 classifier 层生成分类预测
        classification_output = self.classifier(cross_attention_output)
        
        return classification_output, image_first_vector, image_second_vector, image_global_vector, image_cls, \
               text_first_vector, text_second_vector, sentence_vector
