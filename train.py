import torch
import torch.nn as nn

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        
        text = nn.functional.one_hot(label, num_classes=10).float().to(device)
        
        optimizer.zero_grad()
        img_cls, txt_cls, img_vec, txt_vec = model(image, text)
        
        similarity = torch.matmul(img_cls, txt_cls.t())
        labels = torch.arange(image.size(0)).to(device)
        loss = criterion(similarity, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
