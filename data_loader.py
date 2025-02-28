import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from transformers import BertTokenizer
from label_to_text import get_text_descriptions
import os
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from torch.utils.data import Dataset
from enum import Enum
import logging
from config import CIFAR10Config, FashionMNISTConfig

# 配置日志
logger = logging.getLogger(__name__)

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

# 新增数据集类型枚举
class DatasetType(Enum):
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"

class DatasetManager:
    """统一数据集管理类"""
    
    def __init__(self, dataset_type: DatasetType, config: DatasetConfig, batch_size=128, num_workers=4):
        self.dataset_type = dataset_type
        self.config = config
        self.batch_size = batch_size
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self._setup_transforms()
        
    def _setup_transforms(self):
        """根据配置初始化数据增强"""
        # 确保配置中的image_size是有效的
        if not hasattr(self.config, 'image_size') or not self.config.image_size:
            image_size = (32, 32)
        else:
            image_size = self.config.image_size
            
        # 确保配置中的mean和std是有效的
        if not hasattr(self.config, 'mean') or not self.config.mean:
            if self.dataset_type == DatasetType.FASHION_MNIST:
                mean = (0.2860,)
                std = (0.3530,)
            else:  # CIFAR10或默认值
                # 使用更标准的归一化值以提高稳定性
                mean = (0.4914, 0.4822, 0.4465)
                std = (0.2023, 0.1994, 0.2010)  # 降低std值，减小数据范围，提高训练稳定性
        else:
            mean = self.config.mean
            std = self.config.std
            
        # 基础数据转换
        self.base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # 增强的数据增强策略
        if self.dataset_type == DatasetType.CIFAR10:
            # CIFAR10的增强数据增强
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                ], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.2)
            ])
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            # Fashion-MNIST的增强数据增强
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.2)
            ])
        else:
            # 默认数据增强
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                self.base_transform
            ])

    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """获取训练集和测试集"""
        # 根据数据集类型获取数据集类
        if self.dataset_type == DatasetType.CIFAR10:
            dataset_class = torchvision.datasets.CIFAR10
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            dataset_class = torchvision.datasets.FashionMNIST
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
            
        train_dataset = dataset_class(
            root='./data', train=True, download=True, transform=self.train_transform)
        test_dataset = dataset_class(
            root='./data', train=False, download=True, transform=self.base_transform)
        return train_dataset, test_dataset
        
    def get_model_paths(self, dataset_name: str) -> Dict[str, str]:
        """获取模型保存路径"""
        return {
            'checkpoint': f'./checkpoints/{dataset_name}_model.pth',
            'best_model': f'./checkpoints/{dataset_name}_best_model.pth'
        }
        
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

def get_dataset_and_loaders(dataset_name: str = 'cifar10', 
                          batch_size: int = 128) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """获取数据集和数据加载器的便捷函数
    
    Args:
        dataset_name: 数据集名称，'cifar10'或'fashion_mnist'
        batch_size: 批量大小
        
    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        dataset_info: 数据集信息字典
    """
    # 根据数据集名称获取配置和数据集类型
    if dataset_name.lower() == 'cifar10':
        config = CIFAR10Config()
        dataset_type = DatasetType.CIFAR10
    elif dataset_name.lower() in ['fashion_mnist', 'fashion-mnist']:
        config = FashionMNISTConfig()
        dataset_type = DatasetType.FASHION_MNIST
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 使用DatasetManager管理数据集
    manager = DatasetManager(dataset_type, config, batch_size=batch_size)
    
    # 获取数据加载器和数据集信息
    train_loader, test_loader, dataset_info = manager.get_data_loaders()
    
    # 添加img_size信息到dataset_info中
    dataset_info['img_size'] = config.image_size[0]
    
    return train_loader, test_loader, dataset_info

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
    
    # 获取文本描述
    class_descriptions = get_text_descriptions(dataset_name)
    
    # 将标签转换为文本描述
    texts = [class_descriptions[label.item()] for label in labels]
    
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