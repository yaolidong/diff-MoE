import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from transformers import BertTokenizer
from label_to_text import fashion_mnist_label_to_text, cifar10_label_to_text
import os
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from torch.utils.data import Dataset

@dataclass
class DatasetConfig:
    """数据集配置类"""
    name: str
    in_channels: int
    class_names: list
    dataset_class: Any
    transform: transforms.Compose = None
    
    def get_transform(self) -> transforms.Compose:
        """获取数据转换"""
        if self.transform:
            return self.transform
            
        if self.in_channels == 1:
            normalize_mean = (0.5,)
            normalize_std = (0.5,)
        else:
            normalize_mean = (0.5, 0.5, 0.5)
            normalize_std = (0.5, 0.5, 0.5)
            
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])

class DatasetManager:
    """数据集管理类"""
    
    DATASETS = {
        'cifar10': DatasetConfig(
            name='cifar10',
            in_channels=3,
            class_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 
                        'dog', 'frog', 'horse', 'ship', 'truck'],
            dataset_class=torchvision.datasets.CIFAR10
        ),
        'fashion_mnist': DatasetConfig(
            name='fashion_mnist',
            in_channels=1,
            class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
            dataset_class=torchvision.datasets.FashionMNIST
        )
    }
    
    @staticmethod
    def get_model_paths(dataset_name: str) -> Dict[str, str]:
        """获取模型相关路径"""
        return {
            'model': f'model/model_{dataset_name}.pth',
            'best_model': f'model/best_model_{dataset_name}.pth',
            'checkpoint': f'model/checkpoint_{dataset_name}.pth'
        }
    
    def __init__(self, dataset_name: str, batch_size: int = 128, num_workers: int = None):
        self.config = self.DATASETS.get(dataset_name)
        if not self.config:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        self.batch_size = batch_size
        # 根据系统自动设置num_workers
        if num_workers is None:
            if torch.cuda.is_available():
                self.num_workers = min(4, os.cpu_count())
            else:
                self.num_workers = 0  # 在CPU模式下不使用多进程
        else:
            self.num_workers = num_workers
            
        self.transform = self.config.get_transform()
        
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """获取训练集和测试集"""
        train_dataset = self.config.dataset_class(
            root='./data', train=True, download=True, transform=self.transform)
        test_dataset = self.config.dataset_class(
            root='./data', train=False, download=True, transform=self.transform)
        return train_dataset, test_dataset
        
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """获取数据加载器和数据集信息"""
        train_dataset, test_dataset = self.get_datasets()
        
        # 设置生成器
        if torch.backends.mps.is_available():
            generator = torch.Generator(device='cpu')  # MPS设备使用CPU生成器
        else:
            generator = torch.Generator()
        generator.manual_seed(42)  # 设置随机种子
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            generator=generator,
            persistent_workers=self.num_workers > 0
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers,
            generator=generator,
            persistent_workers=self.num_workers > 0
        )
        
        dataset_info = {
            'in_channels': self.config.in_channels,
            'class_names': self.config.class_names,
            'dataset_name': self.config.name,
            'model_paths': self.get_model_paths(self.config.name)
        }
        
        return train_loader, test_loader, dataset_info

def get_dataset_and_loaders(dataset_choice: str = 'fashion_mnist', 
                           batch_size: int = 128) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """获取数据集和数据加载器的便捷函数"""
    manager = DatasetManager(dataset_choice, batch_size)
    return manager.get_data_loaders() 

def general_collate_fn(batch, dataset_name: str):
    """
    通用的数据批处理函数
    
    Args:
        batch: 数据批次
        dataset_name: str, 数据集名称
        
    Returns:
        tuple: (images, input_ids, attention_mask, labels)
    """
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # 根据数据集选择对应的标签转换函数
    label_to_text_fn = {
        'fashion_mnist': fashion_mnist_label_to_text,
        'cifar10': cifar10_label_to_text
    }.get(dataset_name)
    
    if label_to_text_fn is None:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    texts = [label_to_text_fn(label.item()) for label in labels]
    
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", 
                                                local_files_only=True,
                                                cache_dir='./bert_cache')
    except:
        print("正在下载BERT tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                cache_dir='./bert_cache')
    
    text_tensors = tokenizer(
        texts, 
        padding='max_length', 
        truncation=True, 
        max_length=16, 
        return_tensors="pt"
    )
    
    return images, text_tensors['input_ids'], text_tensors['attention_mask'], labels

def fashion_collate_fn(batch):
    """Fashion-MNIST数据集的批处理函数"""
    return general_collate_fn(batch, 'fashion_mnist')

def cifar_collate_fn(batch):
    """CIFAR-10数据集的批处理函数"""
    return general_collate_fn(batch, 'cifar10') 