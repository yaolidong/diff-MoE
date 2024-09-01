import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import label_to_text


def calculate_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(images, input_ids, attention_mask)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = calculate_accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
    return accuracy

def visualize_predictions(model, dataloader, device):
    model.eval()
    images, input_ids, attention_mask, labels = next(iter(dataloader))
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask)
        _, preds = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].cpu().squeeze()
            if img.dim() == 1:
                img = img.view(28, 28)  # 假设图像是 28x28
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Pred: {preds[i].item()}, True: {labels[i].item()}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    print("\nVisualization saved as 'predictions.png'")
