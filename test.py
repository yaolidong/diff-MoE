import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loader import label_to_text

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for image, label in dataloader:
            image = image.to(device)
            label = label.to(device)
            
            text = nn.functional.one_hot(torch.arange(10), num_classes=10).float().to(device)
            
            img_cls, txt_cls, _, _ = model(image, text)
            
            similarity = torch.matmul(img_cls, txt_cls.t())
            _, predicted = similarity.max(1)
            
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    accuracy = correct / total
    return accuracy

def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    text = nn.functional.one_hot(torch.arange(10), num_classes=10).float().to(device)
    
    with torch.no_grad():
        img_cls, txt_cls, _, _ = model(images, text)
        similarity = torch.matmul(img_cls, txt_cls.t())
        _, predicted = similarity.max(1)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().view(28, 28), cmap='gray')
        axes[i].set_title(f"True: {label_to_text[labels[i].item()]}\nPred: {label_to_text[predicted[i].item()]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
