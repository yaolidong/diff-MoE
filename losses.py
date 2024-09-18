import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07, batch_size=64, n_views=1, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.n_views = n_views
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        batch_size = image_features.shape[0]

        # 构建标签
        labels = torch.arange(batch_size, device=self.device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(image_features, text_features.T)

        # 去除对角线元素
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        labels = labels[~mask].view(batch_size, -1)
        similarity_matrix = similarity_matrix[~mask].view(batch_size, -1)

        # 选择正样本和负样本
        positives = similarity_matrix[labels.bool()].view(batch_size, -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # 拼接正负样本
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 计算损失
        logits = logits / self.temperature
        loss = self.criterion(logits, labels)

        return loss
