import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import platform
import timm  # Assuming this import is necessary for the change
import torchvision
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 修改为单通道的均值和标准差
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        in_features = self.vit.head.in_features  # 修改这里
        self.vit.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.vit(x)
    


def train(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return all_preds, all_labels

def visualize_predictions(images, labels, preds, class_names):
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes = axes.flatten()
    for img, lbl, pred, ax in zip(images, labels, preds, axes):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
        ax.set_title(f"True: {class_names[lbl]}\nPred: {class_names[pred]}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    if platform.system() == 'Darwin':  # macOS
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:  # Windows or other
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders()

    model = ViTModel(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 1
    train(model, train_loader, optimizer, device, num_epochs)

    preds, labels = test(model, test_loader, device)

    # 可视化部分测试结果
    class_names = test_loader.dataset.classes
    images, _ = next(iter(test_loader))
    images = images[:9]
    visualize_predictions(images, labels[:9], preds[:9], class_names)

if __name__ == "__main__":
    main()