import torch
import torch.nn.functional as F

def contrastive_loss(features1, features2, temperature=0.5):
    batch_size = features1.shape[0]
    print(f"features的维度: {features.shape}")
    features = F.normalize(features, dim=1)
    print(f"features的维度: {features.shape}")
    similarity_matrix = torch.matmul(features1, features2.T)
    mask = torch.eye(batch_size * 2, dtype=torch.bool, device=features.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)
    
    positives = torch.cat([similarity_matrix[batch_size:, :batch_size], similarity_matrix[:batch_size, batch_size:]], dim=0)
    negatives = similarity_matrix.view(-1)
    
    logits = torch.cat([positives, negatives], dim=0)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    labels[:batch_size * 2] = 1
    
    return F.cross_entropy(logits / temperature, labels)

def compute_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)
