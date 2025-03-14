import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import get_text_descriptions, Flickr8kDataset, TextEnhancedDataset
import os
from typing import Tuple, Dict, Any
from enum import Enum
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig, DatasetConfig

class DatasetType(Enum):
    """数据集类型枚举"""
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"
    FLICKR8K = "flickr8k"

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, dataset_type: DatasetType, config: DatasetConfig, batch_size=128):
        """初始化数据集管理器
        
        Args:
            dataset_type: 数据集类型
            config: 数据集配置
            batch_size: 批大小
        """
        self.dataset_type = dataset_type
        self.config = config
        self.batch_size = batch_size
        self.num_workers = min(8, os.cpu_count() or 2)
        
        # 设置数据转换
        self._setup_transforms()
        
    def _setup_transforms(self):
        """设置数据转换"""
        # 设置归一化参数
        if self.config.in_channels == 1:
            normalize_mean = self.config.mean if hasattr(self.config, 'mean') else (0.5,)
            normalize_std = self.config.std if hasattr(self.config, 'std') else (0.5,)
        else:
            normalize_mean = self.config.mean if hasattr(self.config, 'mean') else (0.5, 0.5, 0.5)
            normalize_std = self.config.std if hasattr(self.config, 'std') else (0.5, 0.5, 0.5)
        
        # 设置图像大小
        if hasattr(self.config, 'image_size'):
            if isinstance(self.config.image_size, int):
                image_size = (self.config.image_size, self.config.image_size)
            else:
                image_size = self.config.image_size
        else:
            image_size = (32, 32)
        
        # 创建训练和测试转换
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        
        print(f"预期输出张量形状: [{self.config.in_channels}, {image_size[0]}, {image_size[1]}]")
    
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """获取训练集和测试集
        
        Returns:
            训练集、验证集和测试集
        """
        if self.dataset_type == DatasetType.CIFAR10:
            # 获取CIFAR10原始数据集
            train_dataset_raw = torchvision.datasets.CIFAR10(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.train_transform
            )
            # 分割训练集为训练集和验证集
            train_size = int(0.8 * len(train_dataset_raw))
            val_size = len(train_dataset_raw) - train_size
            train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(train_dataset_raw, [train_size, val_size])
            
            test_dataset_raw = torchvision.datasets.CIFAR10(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=self.test_transform
            )
            
            # 获取CIFAR10文本描述
            text_descriptions = get_text_descriptions('cifar10')
            
            # 使用TextEnhancedDataset包装原始数据集
            train_dataset = TextEnhancedDataset(
                train_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
            val_dataset = TextEnhancedDataset(
                val_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
            test_dataset = TextEnhancedDataset(
                test_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            # 获取FashionMNIST原始数据集
            train_dataset_raw = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir,
                train=True,
                download=True,
                transform=self.train_transform
            )
            # 分割训练集为训练集和验证集
            train_size = int(0.8 * len(train_dataset_raw))
            val_size = len(train_dataset_raw) - train_size
            train_dataset_raw, val_dataset_raw = torch.utils.data.random_split(train_dataset_raw, [train_size, val_size])
            
            test_dataset_raw = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir,
                train=False,
                download=True,
                transform=self.test_transform
            )
            
            # 获取FashionMNIST文本描述
            text_descriptions = get_text_descriptions('fashion_mnist')
            
            # 使用TextEnhancedDataset包装原始数据集
            train_dataset = TextEnhancedDataset(
                train_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
            val_dataset = TextEnhancedDataset(
                val_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
            test_dataset = TextEnhancedDataset(
                test_dataset_raw, 
                text_descriptions, 
                max_length=self.get_dataset_info().get('max_text_len', 77)
            )
            
        elif self.dataset_type == DatasetType.FLICKR8K:
            # Flickr8k已经是多模态数据集，无需包装
            train_dataset = Flickr8kDataset(
                root=self.config.data_dir,
                split='train',
                transform=self.train_transform
            )
            val_dataset = Flickr8kDataset(
                root=self.config.data_dir,
                split='val',
                transform=self.test_transform
            )
            test_dataset = Flickr8kDataset(
                root=self.config.data_dir,
                split='test',
                transform=self.test_transform
            )
            
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
            
        return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器
        
        Returns:
            训练集、验证集和测试集的数据加载器
        """
        train_dataset, val_dataset, test_dataset = self.get_datasets()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息
        
        Returns:
            包含数据集信息的字典
        """
        img_size = (
            self.config.image_size 
            if isinstance(self.config.image_size, (tuple, list)) 
            else (self.config.image_size, self.config.image_size)
        )
        return {
            'in_channels': self.config.in_channels,
            'img_size': img_size[0],  # 现在确保是元组
            'class_names': self.config.class_names,
            'num_classes': len(self.config.class_names),
            'name': self.config.name,
            'patch_size': self.config.patch_size,
            'max_text_len': 77 if self.dataset_type == DatasetType.FLICKR8K else 32,
            'text_embed_dim': 192 if self.dataset_type == DatasetType.FLICKR8K else 128
        } 