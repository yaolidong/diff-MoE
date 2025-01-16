"""配置文件，存储所有配置参数"""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    label_smoothing: float = 0.1
    early_stopping_patience: int = 10
    warmup_epochs: int = 5

@dataclass
class VisualizationConfig:
    """可视化配置"""
    dpi: int = 300
    figsize: tuple = (10, 6)
    cmap: str = 'viridis'
    save_dir: str = 'visualizations' 