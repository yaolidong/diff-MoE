import os
import sys
import time
import datetime  # 添加 datetime 模块导入
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
from data_loader import DatasetManager, DatasetType
from model import MultiModalMoE
from train import train
from datasets import (
    get_text_descriptions,
    CIFAR10_DESCRIPTIONS,
    FASHION_MNIST_DESCRIPTIONS,
    FLICKR8K_DESCRIPTIONS
)
from utils import setup_environment, plot_training_curves, print_model_summary, plot_confusion_matrix, visualize_predictions, visualize_expert_regions, visualize_expert_activations
from packaging import version  # 添加版本比较支持
from test import evaluate  # 修改导入语句
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig, GlobalConfig, TrainingConfig, ModelConfig

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# 完全禁用PyTorch编译优化
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

def create_model(dataset_info, config: ModelConfig, device):
    """创建模型
    
    Args:
        dataset_info: 数据集信息
        config: 模型配置
        device: 运行设备
        
    Returns:
        创建的模型
    """
    # 从数据集信息中获取参数
    in_channels = dataset_info['in_channels']
    img_size = dataset_info['img_size']
    num_classes = dataset_info['num_classes']
    patch_size = dataset_info.get('patch_size', 4)
    max_text_len = dataset_info.get('max_text_len', 32)
    text_embed_dim = dataset_info.get('text_embed_dim', 128)
    
    # 创建模型
    model = MultiModalMoE(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embed_dim=config.embed_dim,
        num_shared_experts=config.num_shared_experts,
        num_modality_specific_experts=config.num_modality_specific_experts,
        top_k=config.top_k,
        dropout=config.dropout,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        capacity_factor=config.capacity_factor,
        device=device,  # 直接传递device对象
        vocab_size=config.vocab_size,
        max_text_len=max_text_len,
        text_embed_dim=text_embed_dim
    )
    
    # 打印模型信息
    logging.info(f"创建模型: MultiModalMoE")
    logging.info(f"输入通道数: {in_channels}, 图像大小: {img_size}x{img_size}, 类别数: {num_classes}")
    logging.info(f"Patch大小: {patch_size}")
    logging.info(f"嵌入维度: {config.embed_dim}, 共享专家数: {config.num_shared_experts}, 模态特定专家数: {config.num_modality_specific_experts}")
    logging.info(f"Top-K: {config.top_k}, Dropout: {config.dropout}, 注意力头数: {config.num_heads}, 层数: {config.num_layers}")
    logging.info(f"专家容量因子: {config.capacity_factor}")
    logging.info(f"设备: {device}")  # 添加设备信息的日志
    
    return model.to(device)

def load_checkpoint(model, checkpoint_path, device):
    """加载检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 运行设备
        
    Returns:
        加载了检查点的模型和最佳验证准确率
    """
    if not os.path.exists(checkpoint_path):
        logging.warning(f"检查点文件不存在: {checkpoint_path}")
        return model, 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0)
        logging.info(f"成功加载检查点，最佳验证准确率: {best_val_acc:.2f}%")
        return model, best_val_acc
    except Exception as e:
        logging.error(f"加载检查点时出错: {str(e)}")
        return model, 0

def main(args):
    """主函数
    
    Args:
        args: 命令行参数
    """
    # 创建全局配置
    global_config = GlobalConfig()
    global_config.debug = args.debug
    
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG if global_config.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 初始化环境
    logging.info("初始化环境...")
    device = setup_environment(global_config)
    logging.info(f"使用设备: {device}")
    
    # 根据数据集名称选择配置
    if args.dataset.lower() == 'cifar10':
        dataset_config = CIFAR10Config()
        dataset_type = DatasetType.CIFAR10
        class_descriptions = CIFAR10_DESCRIPTIONS
    elif args.dataset.lower() == 'fashion_mnist':
        dataset_config = FashionMNISTConfig()
        dataset_type = DatasetType.FASHION_MNIST
        class_descriptions = FASHION_MNIST_DESCRIPTIONS
    elif args.dataset.lower() == 'flickr8k':
        dataset_config = Flickr8kConfig()
        dataset_type = DatasetType.FLICKR8K
        class_descriptions = FLICKR8K_DESCRIPTIONS
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")
    
    # 创建训练配置
    training_config = TrainingConfig()
    training_config.num_epochs = args.num_epochs
    training_config.batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.weight_decay = args.weight_decay
    
    # 将训练配置添加到全局配置
    global_config.training = training_config
    
    # 创建模型配置
    model_config = ModelConfig()
    
    # 创建数据集管理器
    dataset_manager = DatasetManager(
        dataset_type=dataset_type, 
        config=dataset_config, 
        batch_size=args.batch_size
    )
    train_loader, val_loader, test_loader = dataset_manager.get_data_loaders()
    dataset_info = dataset_manager.get_dataset_info()
    logging.info(f"数据集初始化完成: {dataset_info['name']}")
    
    # 创建模型
    model = create_model(dataset_info, model_config, device)
    
    # 设置保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模式
        logging.info("开始训练模式...")
        
        # 创建优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs
        )
        
        # 训练模型
        model, metrics = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=global_config,
            save_path=save_path,
            class_descriptions=class_descriptions,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # 绘制训练曲线
        plot_path = os.path.join(save_dir, "training_curves.png")
        plot_training_curves(metrics, save_path=plot_path)
        logging.info(f"训练曲线已保存到: {plot_path}")
        
        # 在测试集上评估
        logging.info("在测试集上评估模型...")
        test_results = evaluate(model, test_loader, device, class_names=dataset_info['class_names'], class_descriptions=class_descriptions)
        print_model_summary(model, test_results, dataset_info['class_names'])
        
    elif args.mode == 'test':
        # 测试模式
        logging.info("开始测试模式...")
        
        # 加载检查点
        checkpoint_path = args.checkpoint_path
        if not checkpoint_path:
            checkpoint_path = training_config.checkpoint_path
        
        model, _ = load_checkpoint(model, checkpoint_path, device)
        
        # 在测试集上评估
        test_results = evaluate(model, test_loader, device, class_names=dataset_info['class_names'], class_descriptions=class_descriptions)
        print_model_summary(model, test_results, dataset_info['class_names'])
    
    else:
        raise ValueError(f"不支持的模式: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态混合专家模型训练和测试")
    parser.add_argument("--dataset", type=str, default="cifar10", help="数据集名称: cifar10, fashion_mnist, flickr8k")
    parser.add_argument("--mode", type=str, default="train", help="运行模式: train, test")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="检查点路径")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--debug", action="store_true", help="是否启用调试模式")
    
    args = parser.parse_args()
    main(args) 
