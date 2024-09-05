import torch
import torch.nn.functional as F

def contrastive_loss(features1, features2, temperature=0.5):
    batch_size = features1.shape[0]
    features = torch.cat([features1, features2], dim=0)
    
    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)
    
    # 创建标签
    labels = torch.arange(batch_size, device=features.device)
    labels = torch.cat([labels, labels], dim=0)
    
    # 创建掩码以排除自身比较
    mask = torch.eye(2*batch_size, dtype=torch.bool, device=features.device)
    
    # 应用掩码，但保持原始形状
    similarity_matrix_masked = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # 创建正例和负例的标签
    pos_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # 提取正例和负例
    positives = similarity_matrix_masked[pos_label & (~mask)].view(2*batch_size, -1)
    negatives = similarity_matrix_masked[~pos_label].view(2*batch_size, -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2*batch_size, dtype=torch.long, device=features.device)
    
    return F.cross_entropy(logits / temperature, labels)

def compute_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)
