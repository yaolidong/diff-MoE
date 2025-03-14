import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Any, Tuple
from test import evaluate
from utils import setup_environment, plot_training_curves
# 添加Profiler相关导入
from torch.profiler import profile, record_function, ProfilerActivity
import datetime
from datasets import CIFAR10_DESCRIPTIONS, FASHION_MNIST_DESCRIPTIONS, FLICKR8K_DESCRIPTIONS
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from config import GlobalConfig, DeviceConfig, VisualizationConfig
import torch.nn.functional as F


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
    metrics = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
        'lr': [], 'router_z_loss': [], 'router_balance_loss': [], 'cross_modal_loss': [],
        'contrastive_loss': []
    }
    start_time = time.time()
    
    print(f"\n{config.device}")  # 打印设备信息
    print(f"批大小: {train_loader.batch_size}")
    print(f"训练轮数: {config.training.num_epochs}")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    try:
        for epoch in range(config.training.num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_total_loss = 0
            train_router_z_loss = 0
            train_router_balance_loss = 0
            train_cross_modal_loss = 0
            train_contrastive_loss = 0
            train_correct = 0
            train_total = 0
            valid_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # 数据加载和预处理
                    if isinstance(batch, (list, tuple)) and len(batch) == 4:  # TextEnhancedDataset或多模态输入
                        images, input_ids, attention_mask, labels = batch
                        
                        # 确保所有数据都在正确的设备上
                        images = images.to(device)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)
                        
                        # 前向传播
                        outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
                    elif isinstance(batch, (list, tuple)) and len(batch) == 3:  # Flickr8k类型的输入
                        images, captions, labels = batch
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        if isinstance(captions, torch.Tensor) and captions.dim() == 2:
                            # 已经是token化的文本
                            text_tokens = captions.to(device)
                            attention_mask = torch.ones_like(text_tokens).to(device)
                        else:
                            # 需要处理文本
                            # 如果没有传入tokenizer，这里可能会出错
                            print("警告：收到原始文本，但未实现处理方法，跳过批次")
                            continue
                            
                        outputs = model(images, text_tokens=text_tokens, attention_mask=attention_mask)
                    else:  # 只有图像输入
                        images, labels = batch
                        images = images.to(device)
                        labels = labels.to(device)
                            
                        outputs = model(images)
                    
                    # 计算损失
                    logits = outputs['logits']
                    router_loss = outputs.get('router_loss', 0)
                    router_z_loss = outputs.get('router_z_loss', 0)
                    router_balance_loss = outputs.get('router_balance_loss', 0)
                    cross_modal_loss = outputs.get('cross_modal_loss', 0)
                    contrastive_loss = outputs.get('contrastive_loss', 0)
                    
                    # 添加详细的训练日志
                    if batch_idx % 100 == 0:  # 每100个批次打印一次
                        print("\n" + "="*50)
                        print(f"Epoch {epoch}, Batch {batch_idx}")
                        print(f"Logits shape: {logits.shape}")
                        print(f"Logits mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                        print(f"Logits min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
                        
                        # 打印每个类别的logits平均值
                        class_means = logits.mean(dim=0)
                        print("\nPer-class logits means:")
                        for i, mean in enumerate(class_means):
                            print(f"Class {i}: {mean.item():.4f}")
                        
                        # 计算并打印预测分布
                        _, predicted = torch.max(logits.data, 1)
                        pred_dist = torch.bincount(predicted, minlength=logits.size(1))
                        print("\nPrediction distribution:")
                        for i, count in enumerate(pred_dist):
                            print(f"Class {i}: {count.item()} samples")
                            
                        # 打印真实标签分布
                        label_dist = torch.bincount(labels, minlength=logits.size(1))
                        print("\nTrue label distribution:")
                        for i, count in enumerate(label_dist):
                            print(f"Class {i}: {count.item()} samples")
                        
                        # 打印softmax后的概率分布
                        probs = F.softmax(logits, dim=1)
                        print("\nMean probabilities per class:")
                        mean_probs = probs.mean(dim=0)
                        for i, prob in enumerate(mean_probs):
                            print(f"Class {i}: {prob.item():.4f}")
                    
                    # 分类损失 + 路由损失
                    loss = criterion(logits, labels) + router_loss
                    
                    # 反向传播和优化
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 计算准确率
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_total_loss += loss.item()
                    
                    # 记录各种损失
                    if isinstance(router_z_loss, torch.Tensor):
                        train_router_z_loss += router_z_loss.item()
                    else:
                        train_router_z_loss += router_z_loss
                        
                    if isinstance(router_balance_loss, torch.Tensor):
                        train_router_balance_loss += router_balance_loss.item()
                    else:
                        train_router_balance_loss += router_balance_loss
                        
                    if isinstance(cross_modal_loss, torch.Tensor):
                        train_cross_modal_loss += cross_modal_loss.item()
                    else:
                        train_cross_modal_loss += cross_modal_loss
                    
                    if isinstance(contrastive_loss, torch.Tensor):
                        train_contrastive_loss += contrastive_loss.item()
                    else:
                        train_contrastive_loss += contrastive_loss
                    
                    valid_batches += 1
                    
                    # 更新进度条
                    if train_total > 0:
                        progress_bar.set_postfix({
                            'loss': f"{train_total_loss/valid_batches:.4f}",
                            'acc': f"{100.*train_correct/train_total:.2f}%",
                            'r_loss': f"{router_loss:.4f}" if isinstance(router_loss, float) else f"{router_loss.item():.4f}"
                        })
                    
                except Exception as e:
                    print(f"\n处理批次 {batch_idx} 时出错: {str(e)}")
                    if config.debug:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # 计算训练指标
            if valid_batches > 0 and train_total > 0:
                train_loss = train_total_loss / valid_batches
                train_acc = 100. * train_correct / train_total
                train_router_z_loss = train_router_z_loss / valid_batches
                train_router_balance_loss = train_router_balance_loss / valid_batches
                train_cross_modal_loss = train_cross_modal_loss / valid_batches
                train_contrastive_loss = train_contrastive_loss / valid_batches
            else:
                print("\n警告：本轮没有有效的训练样本")
                train_loss = float('inf')
                train_acc = 0.0
                train_router_z_loss = 0.0
                train_router_balance_loss = 0.0
                train_cross_modal_loss = 0.0
                train_contrastive_loss = 0.0
            
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
            metrics['router_z_loss'].append(train_router_z_loss)
            metrics['router_balance_loss'].append(train_router_balance_loss)
            metrics['cross_modal_loss'].append(train_cross_modal_loss)
            metrics['contrastive_loss'].append(train_contrastive_loss)
            
            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs} 完成 ({epoch_time:.2f}s)")
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"路由损失: Z={train_router_z_loss:.6f}, Balance={train_router_balance_loss:.6f}, CrossModal={train_cross_modal_loss:.6f}, Contrastive={train_contrastive_loss:.6f}")
            print(f"学习率: {current_lr:.8f}")
            
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