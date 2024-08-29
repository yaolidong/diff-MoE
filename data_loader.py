import torch
import torchvision
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))  # 展平图像
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

label_to_text = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}
