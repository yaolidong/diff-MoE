import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        # 特征归一化
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T)
        
        # 应用温度缩放
        similarity_matrix = similarity_matrix / self.temperature
        
        # 创建标签矩阵（对角线为1，其他为0）
        labels = torch.eye(features.shape[0], device=features.device)
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        loss = (-log_prob * labels).sum(dim=1).mean()
        
        return loss