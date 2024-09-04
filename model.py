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
        image_first_vector, image_second_vector, image_cls_first, image_cls_second = self.image_tower(images)
        text_first_vector, text_second_vector, text_cls_first, text_cls_second = self.text_tower(input_ids, attention_mask)
        print(f"image_second_vector的维度: {image_second_vector.shape}")
        print(f"text_second_vector的维度: {text_second_vector.shape}")

        # 确保 image_second_vector 和 text_second_vector 具有相同的维度
        if image_second_vector.dim() == 2:
            image_second_vector = image_second_vector.unsqueeze(1)
        if text_second_vector.dim() == 2:
            text_second_vector = text_second_vector.unsqueeze(1)

        cross_attention_output = self.cross_attention(image_second_vector, text_second_vector)  # 确保传入两个参数
        cross_attention_output = cross_attention_output.mean(dim=1)  # 平均池化
        outputs = self.classifier(cross_attention_output)

        return image_first_vector, image_second_vector, image_cls_first, image_cls_second, \
               text_first_vector, text_second_vector, text_cls_first, text_cls_second, outputs
