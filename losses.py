import torch
import torch.nn.functional as F

def contrastive_loss(features1, features2, temperature=0.5):
    batch_size = features1.shape[0]
    features = torch.cat([features1, features2], dim=0)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    similarity_matrix = torch.matmul(features, features.T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits / temperature, labels)

def compute_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)
