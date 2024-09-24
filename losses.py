import torch
import torch.nn.functional as F
from torch import nn

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, visual_features, textual_features, labels):
        # Normalize features
        visual_features = F.normalize(visual_features, dim=1)
        textual_features = F.normalize(textual_features, dim=1)

        # Compute similarity scores
        similarity_matrix = torch.matmul(visual_features, textual_features.T)

        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create labels
        # labels = torch.arange(visual_features.size(0)).cuda()

        # Calculate cross-entropy loss
        loss_v2t = F.cross_entropy(similarity_matrix, labels)
        loss_t2v = F.cross_entropy(similarity_matrix.T, labels)

        # Combine the two losses
        loss = (loss_v2t + loss_t2v) / 2.0

        return loss

