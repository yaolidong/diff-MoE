import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features, labels):
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # 使用传入的 labels
        labels = labels.view(-1)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)

        # 总损失是两个方向损失的平均
        total_loss = (loss_i2t + loss_t2i) / 2

        return total_loss