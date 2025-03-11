import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
from data_loader import get_dataset_and_loaders, DatasetManager, DatasetType, DatasetConfig
from model import MultiModalMoE
from train import train
import test
from datasets import get_text_descriptions
from utils import setup_environment, plot_training_curves, print_model_summary
from packaging import version  # 添加版本比较支持
from test import test_model
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig

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

def create_model(dataset_info, device):
    """创建模型
    
    Args:
        dataset_info: 数据集信息字典
        device: 运行设备
        
    Returns:
        创建的模型
    """
    try:
        logging.info("正在提取数据集信息...")
        in_channels = dataset_info['in_channels']
        img_size = dataset_info.get('img_size', 32)
        patch_size = dataset_info.get('patch_size', 4)  # 新增patch_size
        num_classes = len(dataset_info['class_names'])
        dataset_name = dataset_info.get('name', '').lower()
        text_descriptions = dataset_info.get('text_descriptions', [])  # 添加文本描述获取
        
        logging.info(f"数据集信息: in_channels={in_channels}, img_size={img_size}, num_classes={num_classes}, dataset_name={dataset_name}")
        
        # 检查参数有效性
        if in_channels <= 0:
            raise ValueError(f"输入通道数必须大于0，但得到了{in_channels}")
        if img_size <= 0:
            raise ValueError(f"图像尺寸必须大于0，但得到了{img_size}")
        if num_classes <= 0:
            raise ValueError(f"类别数必须大于0，但得到了{num_classes}")
            
        logging.info("开始创建模型实例...")
        
        # 创建MultiModalMoE模型
        logging.info("创建多模态MoE模型")
        model = MultiModalMoE(
            img_size=img_size,
            patch_size=patch_size,  # 添加patch_size参数
            in_channels=in_channels,
            num_classes=num_classes,
            text_descriptions=text_descriptions,
            num_shared_experts=8,          # 修改参数名
            num_modality_specific_experts=2,
            expert_type='resnet',          # 添加专家类型
            moe_layer='parallel',          # 添加MoE层类型
            use_gating=True,               # 添加门控
            use_attention=True,            # 添加注意力
            device=device
        )
        
        # 打印模型参数统计
        logging.info("开始轻量级参数统计...")
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"总参数量: {total_params:,} (详细统计已跳过)")
        
        # 将模型移动到指定设备
        logging.info(f"开始移动模型到设备: {device}")
        model = model.to(device)
        logging.info(f"模型移动完成")
        
        # 内存检查
        if device.type == 'cuda':
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
            mem_cached = torch.cuda.memory_reserved(device) / 1024**3
            logging.info(f"GPU内存使用: 已分配 {mem_alloc:.2f} GB / 缓存 {mem_cached:.2f} GB")
        
        return model
        
    except Exception as e:
        logging.error(f"创建模型时发生致命错误: {str(e)}", exc_info=True)  # 添加详细异常信息
        raise

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
         use_profiler=False, profile_epochs=None, profile_steps=100, debug=False):
    """主函数
    
    Args:
        dataset_name: 数据集名称
        mode: 运行模式 ('train' 或 'test')
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        checkpoint_path: 检查点路径
        save_dir: 保存目录
        accumulation_steps: 梯度累积步数
        use_profiler: 是否使用性能分析器
        profile_epochs: 要分析的轮数
        profile_steps: 每轮分析的步数
        debug: 是否启用调试模式
    """
    try:
        # 设置日志格式
        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # 初始化环境
        logging.info("初始化环境...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用设备: {device}")
        
        # 内存清理 - 移到device定义之后
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            logging.info(f"清理后GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            logging.info(f"CUDA版本: {torch.version.cuda}")
            logging.info(f"当前CUDA设备: {torch.cuda.current_device()}")
            logging.info(f"CUDA设备数量: {torch.cuda.device_count()}")
            logging.info(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
            
        # 初始化数据集
        logging.info("初始化数据集...")
        
        # 根据数据集名称选择配置
        if dataset_name.lower() == 'cifar10':
            config = CIFAR10Config()
            dataset_type = DatasetType.CIFAR10
        elif dataset_name.lower() == 'fashion_mnist':
            config = FashionMNISTConfig()
            dataset_type = DatasetType.FASHION_MNIST
        elif dataset_name.lower() == 'flickr8k':
            config = Flickr8kConfig()
            dataset_type = DatasetType.FLICKR8K
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
            
        # 创建数据集管理器
        dataset_manager = DatasetManager(dataset_type=dataset_type, config=config, batch_size=batch_size)
        train_loader, val_loader, test_loader = dataset_manager.get_data_loaders()
        dataset_info = dataset_manager.get_dataset_info()
        logging.info(f"数据集初始化完成，数据集信息: {dataset_info}")
        
        # 验证数据加载器
        logging.info("验证数据加载器...")
        try:
            test_batch = next(iter(train_loader))
            logging.info(f"数据加载检查通过，批次形状: {test_batch[0].shape}")
        except StopIteration:
            logging.error("数据加载器为空，请检查数据集路径和配置")
            raise
        except Exception as e:
            logging.error("数据加载失败:", exc_info=True)
            raise
        
        # 创建模型
        logging.info("开始创建模型...")
        model = create_model(dataset_info, device)
        
        # 调试模式设置
        if debug:
            logging.info("调试模式已禁用，以避免可能的栈溢出")
            # 暂时禁用调试模式
            # model.debug_forward = True
        
        # 打印模型结构
        logging.info("\n模型结构:")
        logging.info(str(model))
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"\n模型参数统计:")
        logging.info(f"总参数量: {total_params:,}")
        
        # 检查模型是否在正确的设备上
        for name, param in model.named_parameters():
            if param.device != device:
                logging.warning(f"参数 {name} 不在正确的设备上: {param.device} != {device}")
        
        # 设置优化器和学习率调度器
        logging.info("\n初始化优化器和学习率调度器...")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        # 如果提供了检查点路径，加载检查点
        if checkpoint_path:
            logging.info(f"\n加载检查点: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_acc = checkpoint['best_val_acc']
                logging.info(f"成功加载检查点，从轮次 {start_epoch} 继续训练")
                logging.info(f"最佳验证准确率: {best_val_acc:.4f}")
            except Exception as e:
                logging.error(f"加载检查点时出错: {str(e)}")
                raise
        else:
            start_epoch = 0
            best_val_acc = 0.0
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 在模型创建后添加明确的日志
        logging.info("模型创建完成，准备开始训练...")
        
        # 确保模型进入训练模式
        model.train()
        logging.info("模型已设置为训练模式")
        
        # 明确启动训练循环
        if mode.lower() == 'train':
            logging.info("===== 开始训练流程 =====")
            for epoch in range(num_epochs):
                logging.info(f"开始第 {epoch+1}/{num_epochs} 轮训练")
                
                # 添加训练代码...
                # train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
                
                logging.info(f"第 {epoch+1} 轮训练完成")
        else:
            logging.info("===== 开始测试流程 =====")
            # 测试代码...
            
        logging.info("程序执行完成")
    except Exception as e:
        logging.error(f"主函数执行出错:", exc_info=True)
        raise

if __name__ == '__main__':
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='训练或测试多模态混合专家模型')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--checkpoint', type=str, help='检查点路径')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--use_profiler', action='store_true', help='是否使用性能分析器')
    parser.add_argument('--profile_epochs', type=int, help='要分析的轮数')
    parser.add_argument('--profile_steps', type=int, default=100, help='每轮分析的步数')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    
    args = parser.parse_args()
    
    # 调用主函数
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
        profile_steps=args.profile_steps,
        debug=args.debug
    ) 
