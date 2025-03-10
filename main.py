import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
from data_loader import get_dataset_and_loaders
from model import MultiModalMoE
from train import train
import test
from datasets import get_text_descriptions
from utils import setup_environment, plot_training_curves, print_model_summary
from packaging import version  # 添加版本比较支持

# 完全禁用PyTorch编译优化
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

def create_model(dataset_info, device):
    """创建模型
    
    Args:
        dataset_info: 数据集信息字典
        device: 运行设备
        
    Returns:
        创建的模型
    """
    # 提取数据集信息
    in_channels = dataset_info['in_channels']
    img_size = dataset_info.get('img_size', 32)
    num_classes = len(dataset_info['class_names'])
    dataset_name = dataset_info.get('name', '').lower()
    
    # 为不同数据集选择不同的配置
    if dataset_name and 'cifar10' in dataset_name:
        model = MultiModalMoE(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=512,
            num_shared_experts=4,
            num_modality_specific_experts=2,
            top_k=2,
            dropout=0.1,
            num_heads=8,
            num_layers=6,
            activation='gelu',
            vocab_size=49408,
            max_text_len=32,
            text_embed_dim=128
        ).to(device)
    elif dataset_name and 'fashion' in dataset_name:
        model = MultiModalMoE(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=384,
            num_shared_experts=4,
            num_modality_specific_experts=2,
            top_k=2,
            dropout=0.1,
            num_heads=6,
            num_layers=4,
            activation='gelu',
            vocab_size=49408,
            max_text_len=32,
            text_embed_dim=128
        ).to(device)
    elif dataset_name and 'flickr' in dataset_name:
        model = MultiModalMoE(
            img_size=img_size,
            patch_size=16,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=384,
            num_shared_experts=4,
            num_modality_specific_experts=2,
            top_k=2,
            dropout=0.1,
            num_heads=6,
            num_layers=4,
            activation='gelu',
            vocab_size=49408,
            max_text_len=77,
            text_embed_dim=192
        ).to(device)
    else:
        # 默认配置
        model = MultiModalMoE(
            img_size=img_size,
            patch_size=4,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=512,
            num_shared_experts=4,
            num_modality_specific_experts=2,
            top_k=2,
            dropout=0.1,
            num_heads=8,
            num_layers=6,
            activation='gelu',
            vocab_size=49408,
            max_text_len=32,
            text_embed_dim=128
        ).to(device)
    
    return model

def load_checkpoint(model, checkpoint_path, device):
    """加载模型检查点
    
    Args:
        model: 模型实例
        checkpoint_path: 检查点路径
        device: 运行设备
        
    Returns:
        加载检查点的模型
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        pass
    
    return model

def main(dataset_name, mode='train', batch_size=256, num_epochs=10, learning_rate=0.0005,
         weight_decay=0.01, checkpoint_path=None, save_dir='checkpoints', accumulation_steps=1,
         use_profiler=False, profile_epochs=None, profile_steps=100):
    """
    主函数，根据模式执行训练或测试
    
    Args:
        dataset_name: 数据集名称 ('cifar10', 'fashion_mnist', 'flickr8k')
        mode: 运行模式 ('train', 'test', 'train_test')
        batch_size: 批量大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        checkpoint_path: 检查点路径
        save_dir: 保存目录
        accumulation_steps: 梯度累积步数
        use_profiler: 是否使用PyTorch Profiler进行性能分析
        profile_epochs: 要进行性能分析的轮次列表，默认为[0]
        profile_steps: 每次性能分析的步数
    """
    if profile_epochs is None:
        profile_epochs = [0]  # 默认在第一个epoch进行性能分析

    device = setup_environment()
    
    # 获取数据集
    train_loader, test_loader, dataset_info = get_dataset_and_loaders(
        dataset_name, batch_size=batch_size
    )
    
    # 创建模型
    model = create_model(dataset_info, device)
    
    # 获取类别名称和描述
    class_names = dataset_info['class_names']
    class_descriptions = get_text_descriptions(dataset_name)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建保存路径
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_model.pt")
    
    # 训练模型
    model, metrics = train(
        model=model, 
        train_loader=train_loader, 
        val_loader=test_loader, 
        device=device, 
        save_path=save_path, 
        num_epochs=num_epochs, 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler,
        accumulation_steps=accumulation_steps,
        use_profiler=use_profiler,
        profile_epochs=profile_epochs,
        profile_steps=profile_steps
    )
    
    # 绘制训练曲线
    plot_training_curves(metrics, save_path.replace('.pt', '_training_curves.png'))
    
    # 测试模式
    if mode == 'test' or mode == 'train_test':
        # 如果需要加载检查点
        if checkpoint_path and mode != 'train_test':
            model = load_checkpoint(model, checkpoint_path, device)
        
        # 测试模型
        test_results = test.test(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names
        )
        
        # 打印模型总结
        print_model_summary(model, test_results, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练或测试多模态MoE模型')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'fashion_mnist', 'flickr8k'],
                       help='要使用的数据集')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test', 'train_test'],
                       help='运行模式: train, test, 或 train_test')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮次数')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='保存目录')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='梯度累积步数')
    # 添加Profiler相关参数
    parser.add_argument('--use_profiler', action='store_true',
                       help='是否使用PyTorch Profiler进行性能分析')
    parser.add_argument('--profile_epochs', type=int, nargs='+', default=[0],
                       help='要进行性能分析的轮次列表，如 --profile_epochs 0 5 9 表示在第1、第6和第10轮进行分析')
    parser.add_argument('--profile_steps', type=int, default=100,
                       help='每次性能分析的步数')
                       
    args = parser.parse_args()
    
    main(
        dataset_name=args.dataset,
        mode=args.mode,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        accumulation_steps=args.accumulation_steps,
        use_profiler=args.use_profiler,
        profile_epochs=args.profile_epochs,
        profile_steps=args.profile_steps
    ) 
