"""配置文件，存储所有配置参数"""
from dataclasses import dataclass, field, asdict
import os
from typing import Tuple, List, Dict, Any
import torch

@dataclass
class DeviceConfig:
    """设备配置"""
    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_name = 'CUDA'
            self.device_properties = {
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
            }
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.device_name = 'MPS'
            self.device_properties = {'name': 'Apple Silicon'}
        else:
            self.device = torch.device('cpu')
            self.device_name = 'CPU'
            self.device_properties = {'name': 'CPU'}
            
    def __str__(self) -> str:
        return f"使用设备: {self.device_name} ({self.device_properties['name']})"

@dataclass
class GlobalConfig:
    """全局配置"""
    # 设备配置
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    # 随机种子
    seed: int = 42
    
    # 是否启用调试模式
    debug: bool = False
    
    # 日志配置
    log_dir: str = 'logs'
    log_level: str = 'INFO'
    
    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)

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
        self.patch_size = 4  # 必须与model.py的patch_size匹配
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.data_dir = 'data'
        self.name = "CIFAR10"
        self.image_size = (32, 32)  # 必须与dataset_info的img_size一致
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
    def __init__(self):
        # 基础参数
        self.embed_dim = 768  # 嵌入维度
        self.num_heads = 12  # 注意力头数
        self.num_layers = 12  # Transformer层数
        self.dropout = 0.1  # Dropout率
        self.vocab_size = 49408  # 词汇表大小，匹配CLIP tokenizer
        
        # MoE相关参数
        self.num_shared_experts = 4  # 共享专家数量
        self.num_modality_specific_experts = 2  # 每个模态的专家数量
        self.top_k = 2  # 每个样本选择的专家数量
        self.capacity_factor = 1.5  # 专家容量因子
        
        # 激活函数和归一化
        self.activation = 'gelu'
        self.layer_norm_eps = 1e-5
        
        # 初始化参数
        self.initializer_range = 0.02

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