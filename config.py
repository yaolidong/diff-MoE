"""配置文件，存储所有配置参数"""
from dataclasses import dataclass, field, asdict
import os
from typing import Tuple, List, Dict, Any

@dataclass
class DatasetConfig:
    """数据集通用配置"""
    patch_size: int = 4
    class_names: List[str] = field(default_factory=list)
    data_dir: str = 'data'
    name: str = "base_dataset"
    image_size: Tuple[int, int] = (32, 32)
    mean: Tuple[float, ...] = (0.5,)
    std: Tuple[float, ...] = (0.5,)
    in_channels: int = 3
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = min(4, os.cpu_count() or 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass 
class CIFAR10Config(DatasetConfig):
    """CIFAR-10专用配置"""
    def __post_init__(self):
        self.patch_size = 4 
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.data_dir = 'data'
        self.name = "CIFAR10"
        self.image_size = (32, 32)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.in_channels = 3
        self.num_classes = 10

@dataclass
class FashionMNISTConfig(DatasetConfig):
    """FashionMNIST专用配置"""
    def __post_init__(self):
        self.patch_size = 4
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self.data_dir = 'data'
        self.name = "FashionMNIST"
        self.image_size = (28, 28)
        self.mean = (0.2860,)
        self.std = (0.3530,)
        self.in_channels = 1
        self.num_classes = 10

@dataclass
class Flickr8kConfig(DatasetConfig):
    """Flickr8k专用配置"""
    def __post_init__(self):
        self.patch_size = 4
        # Flickr8k中常见的图像类别/场景
        self.class_names = [
            '人物', '动物', '运动', '自然风景', '城市场景',
            '室内场景', '交通工具', '水域场景'
        ]
        self.data_dir = 'data/Flickr8k'  # Flickr8k数据存储目录
        self.name = "Flickr8k"
        self.image_size = (224, 224)  # 使用更大的图像尺寸
        self.mean = (0.485, 0.456, 0.406)  # ImageNet标准均值
        self.std = (0.229, 0.224, 0.225)  # ImageNet标准方差
        self.in_channels = 3
        self.num_classes = 8  # 假设有8个主要类别
        self.text_available = True  # 标记数据集包含文本描述

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    num_epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    
    # 检查点配置
    checkpoint_dir: str = 'checkpoints'
    best_model_name: str = 'best_model.pth'
    
    @property
    def checkpoint_path(self) -> str:
        """获取检查点保存路径"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, self.best_model_name)

@dataclass
class ModelConfig:
    """模型配置"""
    # 基础参数
    embed_dim: int = 512
    num_shared_experts: int = 4
    num_modality_specific_experts: int = 2
    top_k: int = 2
    dropout: float = 0.2
    num_heads: int = 8
    num_layers: int = 6
    
    # 激活函数和归一化
    activation: str = 'gelu'
    layer_norm_eps: float = 1e-5
    
    # 初始化参数
    initializer_range: float = 0.02
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 保存目录
    save_dir: str = 'visualizations'
    
    # 图像质量
    dpi: int = 150
    
    # 可视化的样本数量
    num_images_grid: int = 16
    num_samples_tokens: int = 5
    
    # 颜色图
    cmap_heatmap: str = 'viridis'
    cmap_attention: str = 'hot'
    
    # 叠加透明度
    overlay_alpha: float = 0.5
    
    # 图形大小
    expert_regions_fig_size: Tuple[int, int] = (15, 5)
    attention_fig_size: Tuple[int, int] = (15, 10) 