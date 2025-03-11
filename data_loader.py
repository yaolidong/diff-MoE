import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from transformers import CLIPTokenizer
from datasets import get_text_descriptions, Flickr8kDataset, _DESCRIPTIONS_CACHE
import os
from typing import Tuple, Dict, Any
from torch.utils.data import Dataset
from enum import Enum
from dataclasses import dataclass
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig
from copy import deepcopy

# 添加全局变量来存储预加载的文本描述
_PRELOADED_DESCRIPTIONS = {}

# 添加全局tokenizer
_GLOBAL_TOKENIZER = None

# 添加tokenized描述缓存
_TOKENIZED_DESCRIPTIONS_CACHE = {}

# 添加全局缓存用于存储预处理后的数据集
_DATASETS_CACHE = {}

# 添加设备缓存，用于加速数据处理
_DEVICE_CACHE = {}

def get_global_tokenizer():
    """获取全局tokenizer单例，避免重复加载"""
    global _GLOBAL_TOKENIZER
    if _GLOBAL_TOKENIZER is None:
        try:
            _GLOBAL_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", 
                                                      local_files_only=True,
                                                      cache_dir='./clip_cache')
        except:
            print("正在下载CLIP tokenizer...")
            _GLOBAL_TOKENIZER = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                                     cache_dir='./clip_cache')
    return _GLOBAL_TOKENIZER

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
            
        # 根据数据集名称选择不同的变换
        if 'cifar10' in self.name.lower():
            # CIFAR10是32x32的彩色图像
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])
        elif 'fashion' in self.name.lower():
            # Fashion-MNIST是28x28的灰度图像
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])
        elif 'flickr' in self.name.lower():
            # Flickr8k是不定尺寸的彩色图像，调整为224x224
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])
        else:
            # 默认变换，保留原始尺寸
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std)
            ])

class DatasetType(Enum):
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"
    FLICKR8K = "flickr8k"

class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, dataset_type: DatasetType, config: DatasetConfig, batch_size=128, num_workers=8):
        """初始化数据集管理器
        
        Args:
            dataset_type: 数据集类型
            config: 数据集配置
            batch_size: 批次大小
            num_workers: 数据加载线程数
        """
        self.dataset_type = dataset_type
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = None
        self.test_transform = None
        self._setup_transforms()
        # 默认设置设备为None，稍后确定
        self.device = None
        
    def _setup_transforms(self):
        """设置数据转换"""
        # 基于配置创建数据转换
        if self.config.in_channels == 1:
            normalize_mean = self.config.mean if hasattr(self.config, 'mean') else (0.5,)
            normalize_std = self.config.std if hasattr(self.config, 'std') else (0.5,)
            print(f"使用单通道归一化参数: mean={normalize_mean}, std={normalize_std}")
        else:
            normalize_mean = self.config.mean if hasattr(self.config, 'mean') else (0.5, 0.5, 0.5)
            normalize_std = self.config.std if hasattr(self.config, 'std') else (0.5, 0.5, 0.5)
            print(f"使用三通道归一化参数: mean={normalize_mean}, std={normalize_std}")
            
        # 获取图像尺寸，确保是元组格式
        if hasattr(self.config, 'image_size'):
            if isinstance(self.config.image_size, int):
                image_size = (self.config.image_size, self.config.image_size)
            else:
                image_size = self.config.image_size
        else:
            image_size = (32, 32)  # 默认尺寸
            
        print(f"设置图像尺寸为: {image_size}")
        
        # 为训练集添加更多数据增强，以提高模型泛化能力
        train_transforms = [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        
        # 根据数据集类型添加特定的增强
        if self.dataset_type == DatasetType.CIFAR10:
            train_transforms.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(15),
            ])
            print("为CIFAR10添加额外的数据增强")
            
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        
        self.train_transform = transforms.Compose(train_transforms)
        print("训练数据转换设置完成")
        
        # 测试集保持简单转换
        self.test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
        print("测试数据转换设置完成")
        
        # 打印转换后的预期输出形状
        print(f"预期输出张量形状: [{self.config.in_channels}, {image_size[0]}, {image_size[1]}]")
        
    def collate_cifar10(self, batch):
        """CIFAR10的collate函数"""
        # 直接在CPU上处理数据，不要在collate中移至GPU
        return general_collate_fn(batch, 'cifar10', None)
        
    def collate_fashion_mnist(self, batch):
        """Fashion MNIST的collate函数"""
        # 直接在CPU上处理数据，不要在collate中移至GPU
        return general_collate_fn(batch, 'fashion_mnist', None)
        
    def collate_flickr8k(self, batch):
        """Flickr8k的collate函数"""
        # 直接在CPU上处理数据，不要在collate中移至GPU
        return general_collate_fn(batch, 'flickr8k', None)
        
    def get_datasets(self) -> Tuple[Dataset, Dataset]:
        """获取训练集和测试集"""
        # 检查缓存
        cache_key = f"{self.dataset_type.value}_datasets"
        if cache_key in _DATASETS_CACHE:
            print(f"从缓存中加载 {self.dataset_type.value} 数据集")
            return _DATASETS_CACHE[cache_key]
        
        if self.dataset_type == DatasetType.CIFAR10:
            # 加载CIFAR10数据集
            train_dataset = torchvision.datasets.CIFAR10(
                root=self.config.data_dir, 
                train=True, 
                download=True, 
                transform=self.train_transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=self.config.data_dir, 
                train=False, 
                download=True, 
                transform=self.test_transform
            )
            
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            # 加载Fashion-MNIST数据集
            train_dataset = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir, 
                train=True, 
                download=True, 
                transform=self.train_transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root=self.config.data_dir, 
                train=False, 
                download=True, 
                transform=self.test_transform
            )
            
        elif self.dataset_type == DatasetType.FLICKR8K:
            # 加载Flickr8k数据集
            train_dataset = Flickr8kDataset(
                root=self.config.data_dir,
                split='train',
                transform=self.train_transform
            )
            test_dataset = Flickr8kDataset(
                root=self.config.data_dir,
                split='test',
                transform=self.test_transform
            )
            
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
            
        # 预处理小型数据集并缓存到内存
        if self.dataset_type in [DatasetType.CIFAR10, DatasetType.FASHION_MNIST]:
            # 将数据集保存到缓存
            _DATASETS_CACHE[cache_key] = (train_dataset, test_dataset)
            print(f"已缓存 {self.dataset_type.value} 数据集")
            
        return train_dataset, test_dataset
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
        """创建数据加载器"""
        train_dataset, test_dataset = self.get_datasets()
        
        # 检测当前可用设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"数据加载器使用设备: {self.device}")
        
        # 选择合适的collate_fn
        if self.dataset_type == DatasetType.CIFAR10:
            collate_fn = self.collate_cifar10
        elif self.dataset_type == DatasetType.FASHION_MNIST:
            collate_fn = self.collate_fashion_mnist
        elif self.dataset_type == DatasetType.FLICKR8K:
            collate_fn = self.collate_flickr8k
        else:
            collate_fn = None
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # 保持pin_memory=True，但在collate_fn中不移动数据到GPU
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # 保持pin_memory=True，但在collate_fn中不移动数据到GPU
            prefetch_factor=4,
            persistent_workers=True
        )
        
        # 返回数据集信息
        info = {
            'in_channels': self.config.in_channels,
            'img_size': self.config.image_size[0] if hasattr(self.config, 'image_size') else 32,
            'class_names': self.config.class_names,
            'num_classes': len(self.config.class_names),
            'name': self.config.name  # 添加数据集名称
        }
        
        return train_loader, test_loader, info

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息
        
        Returns:
            包含数据集信息的字典
        """
        return {
            'name': self.config.name,
            'in_channels': self.config.in_channels,
            'img_size': self.config.image_size[0] if isinstance(self.config.image_size, tuple) else self.config.image_size,
            'class_names': self.config.class_names,
            'num_classes': len(self.config.class_names)
        }

def get_dataset_and_loaders(dataset_name: str = 'cifar10', 
                          batch_size: int = 128) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    根据数据集名称获取相应的数据加载器
    
    Args:
        dataset_name: 数据集名称
        batch_size: 批量大小，默认128
        
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    num_workers = min(16, os.cpu_count() or 4)  # 使用CPU核心数，最大不超过16
    
    if dataset_name.lower() == 'cifar10':
        config = CIFAR10Config()
        dataset_type = DatasetType.CIFAR10
    elif dataset_name.lower() in ['fashion-mnist', 'fashion_mnist']:
        config = FashionMNISTConfig()
        dataset_type = DatasetType.FASHION_MNIST
    elif dataset_name.lower() == 'flickr8k':
        config = Flickr8kConfig()
        dataset_type = DatasetType.FLICKR8K
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
        
    manager = DatasetManager(
        dataset_type=dataset_type,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    return manager.get_data_loaders()

def general_collate_fn(batch, dataset_name: str, device=None):
    """
    通用数据整理函数，基于数据集名称使用不同的处理逻辑
    使用预处理的tokenized描述提高性能
    
    Args:
        batch: 数据批次
        dataset_name: str, 数据集名称
        device: 可选，目标设备，如果提供则将数据直接移到设备上
        
    Returns:
        处理后的批次数据
    """
    try:
        # 检查当前数据集的缓存
        cache_key = f"{dataset_name}_descriptions"
        if cache_key not in _PRELOADED_DESCRIPTIONS:
            # 从缓存或首次加载
            _PRELOADED_DESCRIPTIONS[cache_key] = get_text_descriptions(dataset_name)
            print(f"已加载{dataset_name}的文本描述")
        
        # Flickr8k数据集的特殊处理
        if dataset_name.lower() == 'flickr8k':
            # 获取图像，文本和标签
            images, captions, labels = zip(*batch)
            
            try:
                # 堆叠图像和转换标签为张量
                images = torch.stack(images)
                labels = torch.tensor(labels)
                print(f"Flickr8k批次数据形状 - 图像: {images.shape}, 标签: {labels.shape}")
                
                # 如果captions是字符串列表，则返回无需tokenize
                if isinstance(captions[0], str):
                    return images, captions, None, labels
                
                # 否则如果已经是tokenized，就直接处理
                if isinstance(captions[0], dict) and 'input_ids' in captions[0]:
                    input_ids = [cap['input_ids'] for cap in captions]
                    attention_mask = [cap['attention_mask'] for cap in captions]
                    input_ids = torch.stack(input_ids)
                    attention_mask = torch.stack(attention_mask)
                    print(f"Tokenized形状 - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
                    return images, input_ids, attention_mask, labels
                
                # 最后情况，是字典但没有input_ids
                caption_tensors = {'input_ids': None}
                return images, caption_tensors['input_ids'], None, labels
                
            except Exception as e:
                print(f"处理Flickr8k批次时出错: {str(e)}")
                raise
        
        # 其他数据集的处理方式
        try:
            images, labels = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(labels)
            print(f"{dataset_name}批次数据形状 - 图像: {images.shape}, 标签: {labels.shape}")
            
            # 检查图像数据范围
            img_min = images.min().item()
            img_max = images.max().item()
            img_mean = images.mean().item()
            img_std = images.std().item()
            print(f"图像统计信息 - 最小值: {img_min:.3f}, 最大值: {img_max:.3f}, 均值: {img_mean:.3f}, 标准差: {img_std:.3f}")
            
            # 获取预先处理的tokenized描述
            tokenized_descriptions = get_tokenized_descriptions(dataset_name)
            
            if not tokenized_descriptions:  # 如果没有可用的tokenized描述
                # 使用预加载的原始文本描述
                descriptions = _PRELOADED_DESCRIPTIONS[dataset_name]
                # 将标签转换为文本描述
                text_inputs = [descriptions.get(label.item(), "") for label in labels]
                print(f"使用原始文本描述，共{len(text_inputs)}个")
                # 返回时不包含text tokens
                return images, None, None, labels
            
            # 收集每个标签对应的tokenized描述
            input_ids_list = []
            attention_mask_list = []
            
            for label in labels:
                label_id = label.item()
                if label_id in tokenized_descriptions:
                    input_ids_list.append(tokenized_descriptions[label_id]['input_ids'])
                    attention_mask_list.append(tokenized_descriptions[label_id]['attention_mask'])
                else:
                    # 使用空张量作为填充
                    input_ids_list.append(torch.zeros(77, dtype=torch.long))
                    attention_mask_list.append(torch.zeros(77, dtype=torch.long))
            
            # 转换为tensor
            input_ids = torch.stack(input_ids_list)
            attention_mask = torch.stack(attention_mask_list)
            print(f"Tokenized形状 - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            
            return images, input_ids, attention_mask, labels
            
        except Exception as e:
            print(f"处理{dataset_name}批次时出错: {str(e)}")
            raise
            
    except Exception as e:
        print(f"collate_fn处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_tokenized_descriptions(dataset_name):
    """
    预先对所有类别描述进行tokenization并缓存结果
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        包含已tokenized描述的字典，键为类别索引
    """
    # 如果已经缓存，直接返回
    if dataset_name in _TOKENIZED_DESCRIPTIONS_CACHE:
        return _TOKENIZED_DESCRIPTIONS_CACHE[dataset_name]
        
    # 获取原始文本描述
    descriptions = get_text_descriptions(dataset_name)
    tokenizer = get_global_tokenizer()
    
    # 将类别ID和描述文本分开准备批处理
    class_ids = list(descriptions.keys())
    texts = list(descriptions.values())
    
    if not texts:  # 如果没有描述，返回空字典
        _TOKENIZED_DESCRIPTIONS_CACHE[dataset_name] = {}
        return {}
    
    # 对所有描述进行批量tokenization
    tokenized = tokenizer(
        texts,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 将结果组织为字典，键为类别索引
    result = {}
    for i, class_id in enumerate(class_ids):
        result[class_id] = {
            'input_ids': tokenized['input_ids'][i],
            'attention_mask': tokenized['attention_mask'][i]
        }
    
    # 缓存结果
    _TOKENIZED_DESCRIPTIONS_CACHE[dataset_name] = result
    
    return result 