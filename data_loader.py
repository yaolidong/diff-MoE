import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import get_text_descriptions, Flickr8kDataset, TextEnhancedDataset
import os
from typing import Tuple, Dict, Any
from enum import Enum
from transformers import AutoTokenizer # Added AutoTokenizer
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig, DatasetConfig, KGAlignmentDatasetConfig # Added KGAlignmentDatasetConfig
from datasets import get_text_descriptions, Flickr8kDataset, TextEnhancedDataset, KGAlignmentDataset # Added KGAlignmentDataset

class DatasetType(Enum):
    """数据集类型枚举"""
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"
    FLICKR8K = "flickr8k"
    KGAlignment = 'kg_alignment' # Added KGAlignment

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
        self.tokenizer = None # Initialize tokenizer attribute

        # 设置通用数据转换 (会被特定数据集逻辑覆盖如果需要)
        self._setup_common_transforms()

        # 初始化数据集实例
        self._initialize_datasets()
        
    def _setup_common_transforms(self):
        """设置通用的数据转换，特定数据集可以在_initialize_datasets中覆盖"""
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

    def _initialize_datasets(self):
        """根据dataset_type初始化train_dataset, val_dataset, test_dataset."""
        # Determine max_text_len from config or provide a default.
        # This helps break the circular dependency with get_dataset_info if it were an issue.
        # For CIFAR10/FashionMNIST, TextEnhancedDataset needs max_length.
        # For Flickr8k, it's handled internally or via its own config.
        # For KGAlignment, it's a param to KGAlignmentDataset.
        default_max_text_len = 32 # A sensible default for non-Flickr8k datasets
        if self.dataset_type == DatasetType.FLICKR8K:
            # Flickr8k might use a different default, e.g. 77, often associated with CLIP
            # This can be fetched from its specific config if available, or hardcoded as a known default.
            # Assuming Flickr8kConfig might have a text_max_len attribute or similar.
            # For now, using a common value for it.
            max_text_len = getattr(self.config, 'text_max_len', 77)
        else:
            max_text_len = getattr(self.config, 'text_max_len', default_max_text_len)


        if self.dataset_type == DatasetType.CIFAR10:
            train_dataset_raw = torchvision.datasets.CIFAR10(
                root=self.config.data_dir, train=True, download=True, transform=self.train_transform)
            train_size = int(0.8 * len(train_dataset_raw))
            val_size = len(train_dataset_raw) - train_size
            train_split_raw, val_split_raw = torch.utils.data.random_split(train_dataset_raw, [train_size, val_size])
            test_dataset_raw = torchvision.datasets.CIFAR10(
                root=self.config.data_dir, train=False, download=True, transform=self.test_transform)
            
            text_descriptions = get_text_descriptions('cifar10')
            self.train_dataset = TextEnhancedDataset(train_split_raw, text_descriptions, max_length=max_text_len)
            self.val_dataset = TextEnhancedDataset(val_split_raw, text_descriptions, max_length=max_text_len)
            self.test_dataset = TextEnhancedDataset(test_dataset_raw, text_descriptions, max_length=max_text_len)
            
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            train_dataset_raw = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir, train=True, download=True, transform=self.train_transform)
            train_size = int(0.8 * len(train_dataset_raw))
            val_size = len(train_dataset_raw) - train_size
            train_split_raw, val_split_raw = torch.utils.data.random_split(train_dataset_raw, [train_size, val_size])
            test_dataset_raw = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir, train=False, download=True, transform=self.test_transform)

            text_descriptions = get_text_descriptions('fashion_mnist')
            self.train_dataset = TextEnhancedDataset(train_split_raw, text_descriptions, max_length=max_text_len)
            self.val_dataset = TextEnhancedDataset(val_split_raw, text_descriptions, max_length=max_text_len)
            self.test_dataset = TextEnhancedDataset(test_dataset_raw, text_descriptions, max_length=max_text_len)
            
        elif self.dataset_type == DatasetType.FLICKR8K:
            # Flickr8k specific transforms might be needed if different from common ones
            # For now, assuming self.train_transform and self.test_transform are appropriate
            # or that Flickr8kDataset handles its own transforms if config provides enough info.
            # If Flickr8kConfig defines specific mean/std/image_size, _setup_common_transforms would use them.
            self.train_dataset = Flickr8kDataset(
                root=self.config.data_dir, split='train', transform=self.train_transform)
            self.val_dataset = Flickr8kDataset(
                root=self.config.data_dir, split='val', transform=self.test_transform)
            self.test_dataset = Flickr8kDataset(
                root=self.config.data_dir, split='test', transform=self.test_transform)

        elif self.dataset_type == DatasetType.KGAlignment:
            if not isinstance(self.config, KGAlignmentDatasetConfig):
                raise TypeError(f"Expected KGAlignmentDatasetConfig, but got {type(self.config)}")

            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            # KGAlignment specific transforms
            kg_train_transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            ])
            kg_test_transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            ])

            # text_tokenizer_max_len for KGAlignmentDataset can also be from config or a default
            kg_text_max_len = getattr(self.config, 'text_tokenizer_max_len', 32)

            self.train_dataset = KGAlignmentDataset(
                config=self.config, split='train', tokenizer=self.tokenizer, transform=kg_train_transform, text_tokenizer_max_len=kg_text_max_len)
            self.val_dataset = KGAlignmentDataset(
                config=self.config, split='val', tokenizer=self.tokenizer, transform=kg_test_transform, text_tokenizer_max_len=kg_text_max_len)
            self.test_dataset = KGAlignmentDataset(
                config=self.config, split='test', tokenizer=self.tokenizer, transform=kg_test_transform, text_tokenizer_max_len=kg_text_max_len)
            
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器
        
        Returns:
            训练集、验证集和测试集的数据加载器
        """
        # Datasets are now expected to be initialized in __init__
        if not all(hasattr(self, ds_attr) for ds_attr in ['train_dataset', 'val_dataset', 'test_dataset']):
            raise RuntimeError("Datasets not initialized. Call _initialize_datasets in __init__.")

        train_loader = DataLoader(
            self.train_dataset,
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
        
        test_loader = DataLoader(
            self.test_dataset,
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

        if self.dataset_type == DatasetType.KGAlignment:
            if not isinstance(self.config, KGAlignmentDatasetConfig):
                raise TypeError(f"Expected KGAlignmentDatasetConfig for KGAlignment, got {type(self.config)}")
            return {
                'name': self.config.name,
                'num_train_samples': len(self.train_dataset) if self.train_dataset else 0,
                'num_val_samples': len(self.val_dataset) if self.val_dataset else 0,
                'num_test_samples': len(self.test_dataset) if self.test_dataset else 0,
                'image_size': self.config.image_size, # Expected to be a tuple (H, W)
                'in_channels': self.config.in_channels,
                'embedding_dim': self.config.embedding_dim,
                # num_classes is not typically relevant for alignment tasks
            }

        # Default info for other dataset types
        img_size_tuple = self.config.image_size if isinstance(self.config.image_size, tuple) else (self.config.image_size, self.config.image_size)

        # Determine max_text_len based on dataset type or config
        if self.dataset_type == DatasetType.FLICKR8K:
            max_text_len = getattr(self.config, 'text_max_len', 77)
        else: # CIFAR10, FashionMNIST
            max_text_len = getattr(self.config, 'text_max_len', 32)

        text_embed_dim_default = 128 # Default for CIFAR10, FashionMNIST
        if self.dataset_type == DatasetType.FLICKR8K:
            text_embed_dim_default = 192 # Default for Flickr8k
        text_embed_dim = getattr(self.config, 'text_embed_dim', text_embed_dim_default)


        return {
            'name': self.config.name,
            'num_train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'num_val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'num_test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'in_channels': self.config.in_channels,
            'img_size': img_size_tuple[0], # Assuming square images or width is primary
            'class_names': self.config.class_names if hasattr(self.config, 'class_names') else [],
            'num_classes': len(self.config.class_names) if hasattr(self.config, 'class_names') else 0,
            'patch_size': self.config.patch_size if hasattr(self.config, 'patch_size') else None,
            'max_text_len': max_text_len,
            'text_embed_dim': text_embed_dim
        }
