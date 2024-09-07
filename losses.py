import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, image_first_vector, image_second_vector, text_first_vector, text_second_vector, labels):
        # 计算图像和文本之间的对比损失
        loss_image_text = self.compute_pairwise_loss(image_second_vector, text_second_vector, labels)
        
        # 计算图像内部的对比损失
        loss_image = self.compute_pairwise_loss(image_first_vector, image_second_vector, labels)
        
        # 计算文本内部的对比损失
        loss_text = self.compute_pairwise_loss(text_first_vector, text_second_vector, labels)
        
        # 总损失是这三个损失的加权和
        total_loss = loss_image_text + 0.5 * loss_image + 0.5 * loss_text
        
        return total_loss

    def compute_pairwise_loss(self, features1, features2, labels):
        batch_size = features1.shape[0]
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(features1, features2, dim=1)
        
        # 计算正例和负例的损失
        positive_loss = (1 - labels) * torch.pow(1 - similarity, 2)
        negative_loss = labels * torch.pow(torch.clamp(similarity - self.margin, min=0.0), 2)
        
        loss = 0.5 * (positive_loss + negative_loss)
        
        return loss.mean() / self.temperature

def compute_loss(outputs, labels):
    return F.cross_entropy(outputs, labels)

