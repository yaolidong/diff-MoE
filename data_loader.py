import torch
from torch.utils.data import DataLoader
import torchvision
from transformers import BertTokenizer
from label_to_text import label_to_text
from image_augmentation import get_augmentation

def get_data_loaders(batch_size=128):
    augmentation = get_augmentation()
    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=None)
    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=None)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=lambda x: collate_fn(x, augmentation))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=lambda x: collate_fn(x, augmentation))
    return train_dataloader, test_dataloader

def collate_fn(batch, augmentation):
    images, labels = zip(*batch)
    
    # 原始图像
    original_images = [augmentation(img) for img in images]
    # print(f"原始图像维度: {[img.shape for img in original_images]}")
    original_images = torch.stack(original_images)
    
    # 增强图像
    augmented_images = [augmentation(img) for img in images]
    # print(f"增强图像维度: {[img.shape for img in augmented_images]}")
    augmented_images = torch.stack(augmented_images)
    
    labels = torch.tensor(labels)
    # for label in labels:
    #     print(label_to_text[label.item()])
    texts = [label_to_text[label.item()] for label in labels]

    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    text_tensors = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    
    input_ids = text_tensors['input_ids']
    attention_mask = text_tensors['attention_mask']
    
    return original_images, input_ids, attention_mask, labels





