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


def train(model, train_loader, val_loader, device, save_path, num_epochs=10, 
          criterion=None, optimizer=None, class_descriptions=None, scheduler=None, accumulation_steps=1,
          use_profiler=False, profile_epochs=[0], profile_steps=100):
    """训练函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        save_path: 保存路径
        num_epochs: 训练轮数
        criterion: 损失函数
        optimizer: 优化器
        class_descriptions: 类别描述
        scheduler: 学习率调度器
        accumulation_steps: 梯度累积步数
        use_profiler: 是否使用PyTorch Profiler
        profile_epochs: 要进行性能分析的轮次列表
        profile_steps: 每次性能分析的步数
        
    Returns:
        训练后的模型和训练指标
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    best_val_acc = 0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start_time = time.time()
    
    print(f"开始训练模型，使用 {device} 设备, 批大小: {train_loader.batch_size}, 累积步数: {accumulation_steps}")
    print(f"实际批大小: {train_loader.batch_size * accumulation_steps}, 共 {num_epochs} 轮")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    try:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            model.train()
            train_total_loss = 0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            optimizer.zero_grad()  # 在每个epoch开始时清零梯度
            
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
                        
                        # 设置当前批次的标签
                        if hasattr(model, 'set_labels'):
                            model.set_labels(labels)
                        
                        # 打印调试信息
                        if i == 0 and epoch == 0:
                            print(f"\n批次数据信息:")
                            print(f"图像形状: {images.shape}, 设备: {images.device}")
                            print(f"文本tokens形状: {input_ids.shape}, 设备: {input_ids.device}")
                            print(f"注意力掩码形状: {attention_mask.shape}, 设备: {attention_mask.device}")
                            print(f"标签形状: {labels.shape}, 设备: {labels.device}")
                        
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
                    
                    # 梯度累积
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    # 更新参数
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    # 计算准确率
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    train_total_loss += loss.item() * accumulation_steps
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f"{train_total_loss/(i+1):.4f}",
                        'acc': f"{100.*train_correct/train_total:.2f}%"
                    })
                    
                except Exception as e:
                    print(f"\n处理批次 {i} 时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 计算训练指标
            train_loss = train_total_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # 验证阶段
            val_loss, val_acc = test.evaluate(model, val_loader, criterion, device)
            
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
            print(f"\nEpoch {epoch+1}/{num_epochs} 完成 ({epoch_time:.2f}s)")
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
        import traceback
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n训练完成，总用时: {total_time:.2f}s")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, metrics