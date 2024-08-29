import torch
import torch.nn as nn
from moe_encoders import ImageMoE, TextMoE

class DualTowerModel(nn.Module):
    def __init__(self, image_input_dim=784, text_input_dim=10, output_dim=128):
        super().__init__()
        self.image_encoder = ImageMoE(input_dim=image_input_dim, output_dim=output_dim)
        self.text_encoder = TextMoE(input_dim=text_input_dim, output_dim=output_dim)
    
    def forward(self, image, text):
        img_cls, img_vec = self.image_encoder(image)
        txt_cls, txt_vec = self.text_encoder(text)
        return img_cls, txt_cls, img_vec, txt_vec
