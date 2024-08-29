import torch
import torch.nn as nn
from moe_encoders import ImageMoE, TextMoE, CrossAttention

class DualTowerModel(nn.Module):
    def __init__(self, num_classes, image_input_dim=784, output_dim=128):
        super().__init__()
        self.image_encoder = ImageMoE(input_dim=image_input_dim, output_dim=output_dim)
        self.text_encoder = nn.Embedding(num_classes, output_dim)
        self.cross_attention = CrossAttention(output_dim)
    
    def forward(self, image, text):
        img_first_vec, img_second_vec, img_cls_first, img_cls_second = self.image_encoder(image)
        txt_first_vec, txt_second_vec, txt_cls_first, txt_cls_second = self.text_encoder(text)
        
        # 确保所有输出都经过归一化
        img_cls_second = nn.functional.normalize(img_cls_second, dim=1)
        txt_cls_second = nn.functional.normalize(txt_cls_second, dim=1)
        
        return img_first_vec, img_second_vec, img_cls_first, img_cls_second, txt_first_vec, txt_second_vec, txt_cls_first, txt_cls_second
