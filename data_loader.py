import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from transformers import BertTokenizer
from label_to_text import label_to_text, cifar10_label_to_text

def get_data_loaders(batch_size=128):
    # Fashion-MNIST的转换
    fashion_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # CIFAR-10的转换
    cifar_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载Fashion-MNIST数据集
    fashion_train = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=fashion_transform)
    fashion_test = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=fashion_transform)

    # 加载CIFAR-10数据集
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=cifar_transform)
    cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=cifar_transform)

    # 创建数据加载器
    fashion_train_loader = DataLoader(fashion_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=fashion_collate_fn)
    fashion_test_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=fashion_collate_fn)
    
    cifar_train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=cifar_collate_fn)
    cifar_test_loader = DataLoader(cifar_test, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=cifar_collate_fn)
    
    return (fashion_train_loader, fashion_test_loader), (cifar_train_loader, cifar_test_loader)

def fashion_collate_fn(batch):
    return general_collate_fn(batch, label_to_text)

def cifar_collate_fn(batch):
    return general_collate_fn(batch, cifar10_label_to_text)

def general_collate_fn(batch, label_dict):
    images, labels = zip(*batch)
    
    # 堆叠图像
    images = torch.stack(images)
    
    labels = torch.tensor(labels)
    texts = [label_dict[label.item()] for label in labels]

    tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased")
    text_tensors = tokenizer(texts, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
    
    input_ids = text_tensors['input_ids']
    attention_mask = text_tensors['attention_mask']
    
    return images, input_ids, attention_mask, labels 