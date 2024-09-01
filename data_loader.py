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

    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    text_tensors = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    
    input_ids = text_tensors['input_ids']
    attention_mask = text_tensors['attention_mask']
    
    # print(f"每个input_ids的维度: {input_ids.shape}")
    # print(f"每个attention_mask的维度: {attention_mask.shape}")

    return images, input_ids, attention_mask, labels





