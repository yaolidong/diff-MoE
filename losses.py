import torch
import torch.nn.functional as F
from torch import nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_features, textual_features):
        # 创建标签矩阵[128,128]
        labels = torch.cat([torch.arange(visual_features.shape[0]) for i in range(2)], dim=0)
        labels = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels = labels.to(visual_features.device)

        combined_features = torch.cat([visual_features, textual_features], dim=0)
        # 归一化
        combined_features = F.normalize(combined_features, dim=1)

        # 计算相似度
        similarity_matrix = torch.matmul(combined_features, combined_features.T)
        similarity_matrix = similarity_matrix / self.temperature

        # 创建掩码矩阵[128,128]
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(similarity_matrix.device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        labels = labels[~mask].view(labels.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(visual_features.device)

        loss = F.cross_entropy(logits, labels)
        return loss 