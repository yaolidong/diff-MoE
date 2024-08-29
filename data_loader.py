import torch
from torch.utils.data import DataLoader
import torchvision
from transformers import BertTokenizer, AutoTokenizer

label_to_text = {
    0: "T-shirt top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

def get_data_loaders(batch_size=128):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: x.view(-1))  # 展平图像
    ])
    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, test_dataloader

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    texts = [label_to_text[label] for label in labels]
    labels = torch.tensor(labels)

    # 将文本直接转换为张量
    text_tensors = torch.tensor([text for text in texts])
    print(f"每个text_tensor的维度: {text_tensors.shape}")
    
    max_length = 128
    if text_tensors.size(1) < max_length:
        padding = torch.zeros((text_tensors.size(0), max_length - text_tensors.size(1)), dtype=torch.float)
        padded_texts = torch.cat((text_tensors, padding), dim=1)
    else:
        padded_texts = text_tensors[:, :max_length]
    
    print(f"每个padded_texts的维度: {padded_texts.shape}")
    return images, padded_texts, labels





