import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Any, Tuple
import test
from utils import setup_environment, plot_training_curves
# 添加Profiler相关导入
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
import logging
from datasets import CIFAR10_DESCRIPTIONS, FASHION_MNIST_DESCRIPTIONS, FLICKR8K_DESCRIPTIONS
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from config import GlobalConfig, DeviceConfig, VisualizationConfig


def train(model, train_loader, val_loader, config: GlobalConfig, save_path: str, 
          criterion=None, optimizer=None, class_descriptions=None, scheduler=None):
    """训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 全局配置对象
        save_path: 保存路径
        criterion: 损失函数
        optimizer: 优化器
        class_descriptions: 类别描述字典
        scheduler: 学习率调度器
        
    Returns:
        训练后的模型和训练指标
    """
    device = config.device.device
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config.training.learning_rate, 
                                    weight_decay=config.training.weight_decay)
    
    # 使用配置中的可视化目录
    vis_config = VisualizationConfig()
    vis_dir = vis_config.save_dir
    
    best_val_acc = 0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start_time = time.time()
    
    print(f"\n{config.device}")  # 打印设备信息
    print(f"批大小: {train_loader.batch_size}")
    print(f"训练轮数: {config.training.num_epochs}")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    try:
        # 验证tokenizer和模型的词汇表大小匹配
        if hasattr(model, 'text_embedding'):
            vocab_size = model.text_embedding.weight.size(0)
            if vocab_size != 49408:
                logging.error(f"词汇表大小配置错误!")
                logging.error(f"当前配置: {vocab_size}")
                logging.error(f"CLIP tokenizer需要: 49408")
                raise ValueError("请将模型的vocab_size设置为49408以匹配CLIP tokenizer")
        
        for epoch in range(config.training.num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_total_loss = 0
            train_correct = 0
            train_total = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
            optimizer.zero_grad()
            
            for i, batch in enumerate(progress_bar):
                try:
                    # 数据加载和预处理
                    if len(batch) == 4:
                        images, input_ids, attention_mask, labels = batch
                        
                        # 确保所有数据都在正确的设备上
                        images = images.to(device)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)
                        
                        # 每10个批次保存样本图像和对应的中文描述
                        if i % 10 == 0:
                            print("\n" + "="*50)
                            print(f"批次 {i} 的样本信息:")
                            
                            # 保存前3个样本的图像和描述
                            for idx in range(min(3, len(labels))):
                                label = labels[idx].item()
                                if class_descriptions and label in class_descriptions:
                                    # 保存图像
                                    img = images[idx].cpu()
                                    # 反归一化图像
                                    img = img * 0.5 + 0.5
                                    
                                    plt.figure(figsize=vis_config.expert_regions_fig_size)
                                    plt.imshow(img.permute(1, 2, 0))
                                    plt.title(f"类别: {class_descriptions[label]}", fontsize=12, fontproperties='SimHei')
                                    plt.axis('off')
                                    
                                    # 保存图像
                                    save_name = f'epoch_{epoch+1}_batch_{i}_sample_{idx}.png'
                                    plt.savefig(os.path.join(vis_dir, save_name), 
                                              bbox_inches='tight', 
                                              dpi=vis_config.dpi)
                                    plt.close()
                                    
                                    print(f"\n样本 {idx}:")
                                    print(f"标签: {label}")
                                    print(f"对应的中文描述: {class_descriptions[label]}")
                                    print(f"已保存图像到: {save_name}")
                                    
                                    if config.debug:  # 在调试模式下打印更多信息
                                        print(f"文本tokens形状: {input_ids[idx].shape}")
                                        print(f"注意力掩码形状: {attention_mask[idx].shape}")
                                        print(f"文本tokens范围: [{input_ids[idx].min().item()}, {input_ids[idx].max().item()}]")
                                        
                                        if hasattr(model, 'text_embedding'):
                                            with torch.no_grad():
                                                text_embed = model.text_embedding(input_ids[idx].unsqueeze(0))
                                                print(f"文本嵌入形状: {text_embed.shape}")
                                                print(f"文本嵌入范围: [{text_embed.min().item():.4f}, {text_embed.max().item():.4f}]")
                            print("="*50 + "\n")
                        
                        # 设置当前批次的标签
                        if hasattr(model, 'set_labels'):
                            model.set_labels(labels)
                        
                        # 前向传播
                        outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
                    else:
                        images, labels = batch
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        if hasattr(model, 'set_labels'):
                            model.set_labels(labels)
                            
                        outputs = model(images)
                    
                    # 计算损失
                    logits = outputs['logits']
                    router_loss = outputs.get('router_loss', 0)
                    loss = criterion(logits, labels) + router_loss
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 计算准确率
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_total_loss += loss.item()
                    valid_batches += 1
                    
                    # 更新进度条
                    if train_total > 0:
                        progress_bar.set_postfix({
                            'loss': f"{train_total_loss/valid_batches:.4f}",
                            'acc': f"{100.*train_correct/train_total:.2f}%"
                        })
                    
                except Exception as e:
                    print(f"\n处理批次 {i} 时出错: {str(e)}")
                    if config.debug:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # 计算训练指标
            if valid_batches > 0 and train_total > 0:
                train_loss = train_total_loss / valid_batches
                train_acc = 100. * train_correct / train_total
            else:
                print("\n警告：本轮没有有效的训练样本")
                train_loss = float('inf')
                train_acc = 0.0
            
            # 验证
            model.eval()
            with torch.no_grad():
                test_results = evaluate(model, val_loader, device, criterion, class_names=class_descriptions)
                val_loss = test_results['loss']
                val_acc = test_results['accuracy']
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # 保存指标
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['lr'].append(current_lr)
            
            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs} 完成 ({epoch_time:.2f}s)")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_acc': best_val_acc,
                    'metrics': metrics
                }, save_path)
                print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")
    
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        if config.debug:
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n训练完成，总用时: {total_time:.2f}s")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, metrics