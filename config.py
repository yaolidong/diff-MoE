"""配置文件，存储所有配置参数"""
from dataclasses import dataclass, field, asdict
import os
from typing import Tuple, List, Dict, Any, Optional, Union
import json
import yaml
import logging

# 配置日志 - 不再重复设置basicConfig
logger = logging.getLogger(__name__)

def load_config_from_env(prefix: str, config_class: Any) -> Dict[str, Any]:
    """从环境变量加载配置，使用指定前缀"""
    config_dict = {}
    for field_name in config_class.__dataclass_fields__:
        env_var_name = f"{prefix}_{field_name}".upper()
        if env_var_name in os.environ:
            # 获取字段类型
            field_type = config_class.__dataclass_fields__[field_name].type
            # 将环境变量转换为对应类型
            try:
                if field_type in (int, float, str, bool):
                    value = os.environ[env_var_name]
                    if field_type == int:
                        config_dict[field_name] = int(value)
                    elif field_type == float:
                        config_dict[field_name] = float(value)
                    elif field_type == bool:
                        config_dict[field_name] = value.lower() in ('true', 'yes', '1', 'y')
                    else:  # str
                        config_dict[field_name] = value
                elif field_type == List[str]:
                    config_dict[field_name] = os.environ[env_var_name].split(',')
                elif field_type == Tuple[float, ...]:
                    config_dict[field_name] = tuple(float(x) for x in os.environ[env_var_name].split(','))
                elif field_type == Tuple[int, int]:
                    values = os.environ[env_var_name].split(',')
                    if len(values) == 2:
                        config_dict[field_name] = (int(values[0]), int(values[1]))
            except (ValueError, TypeError) as e:
                logger.warning(f"无法解析环境变量 {env_var_name}: {e}")
    return config_dict

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return {}
    
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"不支持的配置文件格式: {config_path}")
            return {}
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

@dataclass
class DatasetConfig:
    """数据集通用配置"""
    patch_size: int = 4  # 添加patch_size配置
    class_names: List[str] = field(default_factory=list)  # 添加类别名称字段
    data_dir: str = 'data'  # 确保存在数据目录配置
    name: str = "base_dataset"
    image_size: Tuple[int, int] = (32, 32)
    mean: Tuple[float, ...] = (0.5,)
    std: Tuple[float, ...] = (0.5,)
    in_channels: int = 3
    num_classes: int = 10
    batch_size: int = 128
    num_workers: int = min(4, os.cpu_count() or 2)
    persistent_workers: bool = True
    pin_memory: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "DATASET") -> "DatasetConfig":
        """从环境变量创建配置"""
        config_dict = load_config_from_env(prefix, cls)
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "DatasetConfig":
        """从文件创建配置"""
        config_dict = load_config_from_file(config_path)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass 
class CIFAR10Config(DatasetConfig):
    """CIFAR-10专用配置"""
    def __post_init__(self):
        super().__init__()  # 确保父类初始化
        self.patch_size = 4  # 根据数据集设置具体值
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.data_dir = 'data'  # 显式设置数据目录
        self.name = "CIFAR10"
        self.image_size = (32, 32)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)  # 降低std值，减小数据范围，提高训练稳定性
        self.in_channels = 3
        self.num_classes = 10

@dataclass
class FashionMNISTConfig(DatasetConfig):
    """FashionMNIST专用配置"""
    def __post_init__(self):
        self.patch_size = 4  # 根据数据集设置具体值
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        self.data_dir = 'data'  # 显式设置数据目录
        self.name = "FashionMNIST"
        self.image_size = (28, 28)
        self.mean = (0.2860,)
        self.std = (0.3530,)
        self.in_channels = 1
        self.num_classes = 10

@dataclass
class TrainingConfig:
    """训练配置"""
    # 新增预热比例参数
    warmup_ratio: float = 0.1  # 增加预热比例，提高初期训练稳定性
    # 基本训练参数
    num_epochs: int = 20  # 增加训练轮数，让模型有足够的收敛时间
    batch_size: int = 128  # 减小批次大小，提高优化稳定性
    learning_rate: float = 0.0005  # 显著降低学习率，避免训练不稳定
    weight_decay: float = 0.01  # 减小权重衰减，防止过度正则化
    gradient_clip_val: float = 0.5  # 减小梯度裁剪值，避免大梯度更新
    
    # 损失函数相关
    label_smoothing: float = 0.1
    router_loss_weight: float = 0.01  # 增加路由损失权重，让路由器更好学习
    
    # 早停和预热
    early_stopping_patience: int = 10
    warmup_epochs: int = 2
    
    # 优化器配置
    optimizer: str = 'AdamW'
    scheduler: str = 'cosine_warmup'
    
    # 检查点配置
    checkpoint_dir: str = 'checkpoints'
    best_model_name: str = 'best_model.pth'
    
    # 混合精度和分布式训练
    use_amp: bool = False  # 禁用自动混合精度，减少训练复杂度
    use_gradient_checkpointing: bool = True  # 使用梯度检查点
    grad_accum_steps: int = 1  # 不使用梯度累积，简化训练流程
    use_torch_compile: bool = False  # 控制是否使用torch.compile优化（默认关闭以避免Triton错误）
    
    # 性能优化参数
    cudnn_benchmark: bool = True  # 启用cuDNN基准测试以提高性能
    cudnn_deterministic: bool = False  # 关闭确定性模式以提高性能
    
    # 日志配置
    log_interval: int = 10  # 每多少个batch记录一次日志
    
    @property
    def checkpoint_path(self) -> str:
        """获取检查点保存路径"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, self.best_model_name)
    
    @classmethod
    def from_env(cls, prefix: str = "TRAINING") -> "TrainingConfig":
        """从环境变量创建配置"""
        config_dict = load_config_from_env(prefix, cls)
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "TrainingConfig":
        """从文件创建配置"""
        config_dict = load_config_from_file(config_path)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

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
    
    # 高级功能
    use_gradient_checkpointing: bool = True
    
    @classmethod
    def from_env(cls, prefix: str = "MODEL") -> "ModelConfig":
        """从环境变量创建配置"""
        config_dict = load_config_from_env(prefix, cls)
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "ModelConfig":
        """从文件创建配置"""
        config_dict = load_config_from_file(config_path)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

@dataclass
class VisualizationConfig:
    """可视化相关配置"""
    # 图像数量配置
    num_images_grid: int = 5  # 预测结果网格中显示的图像数量
    num_samples_tokens: int = 5  # token分布可视化的样本数量
    num_samples_router: int = 5  # 路由决策可视化的样本数量
    num_samples_attention: int = 5  # 注意力权重可视化的样本数量
    
    # 图像尺寸配置
    expert_regions_fig_size: Tuple[int, int] = (15, 5)  # 专家区域可视化的图像尺寸
    confusion_matrix_fig_size: Tuple[int, int] = (10, 8)  # 混淆矩阵的图像尺寸
    expert_usage_fig_size: Tuple[int, int] = (10, 6)  # 专家使用率分布的图像尺寸
    single_prediction_fig_size: Tuple[int, int] = (6, 8)  # 单张图像预测结果的图像尺寸
    
    # DPI配置
    dpi: int = 300  # 图像保存时的DPI值
    
    # 颜色配置
    cmap_attention: str = 'hot'  # 注意力图的颜色映射
    cmap_heatmap: str = 'viridis'  # 热力图的颜色映射
    cmap_confusion: str = 'Blues'  # 混淆矩阵的颜色映射
    
    # 透明度配置
    overlay_alpha: float = 0.5  # 叠加图的透明度
    
    # 保存路径配置
    save_dir: str = 'visualizations'  # 可视化结果保存目录 
    
    @classmethod
    def from_env(cls, prefix: str = "VIS") -> "VisualizationConfig":
        """从环境变量创建配置"""
        config_dict = load_config_from_env(prefix, cls)
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, config_path: str) -> "VisualizationConfig":
        """从文件创建配置"""
        config_dict = load_config_from_file(config_path)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

def create_config_file(config_obj: Any, file_path: str, file_format: str = 'yaml'):
    """创建配置文件"""
    config_dict = asdict(config_obj)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if file_format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif file_format == 'yaml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            logger.warning(f"不支持的配置文件格式: {file_format}")
    except Exception as e:
        logger.error(f"创建配置文件失败: {e}")

# 创建默认配置文件（如果不存在）
def create_default_configs():
    """创建默认配置文件"""
    # 创建配置目录
    os.makedirs('configs', exist_ok=True)
    
    # 创建各类配置文件
    create_config_file(TrainingConfig(), 'configs/training.yaml')
    create_config_file(ModelConfig(), 'configs/model.yaml')
    create_config_file(CIFAR10Config(), 'configs/cifar10.yaml')
    create_config_file(FashionMNISTConfig(), 'configs/fashion_mnist.yaml')
    create_config_file(VisualizationConfig(), 'configs/visualization.yaml')
    
    logger.info("已创建默认配置文件")

# 初始化时创建默认配置文件
if not os.path.exists('configs'):
    create_default_configs() 