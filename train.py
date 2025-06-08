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
    
    # if criterion is None: # Criterion will be handled by alignment loss later
    #     criterion = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=config.training.learning_rate, 
                                    weight_decay=config.training.weight_decay)
    
    # 使用配置中的可视化目录
    vis_config = VisualizationConfig()
    vis_dir = vis_config.save_dir
    
    best_val_acc = 0 # This will need to be changed to best_val_loss or similar metric for alignment
    metrics = {
        'train_loss': [],
        # 'train_acc': [], # Removed accuracy
        'val_loss': [],
        # 'val_acc': [], # Removed accuracy
        'train_alignment_loss': [], # Added placeholder
        'val_alignment_loss': [], # Added placeholder
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
            # train_correct = 0 # Accuracy removed
            # train_total = 0 # Accuracy removed
            valid_batches = 0
            train_total_alignment_loss = 0.0 # Initialize total alignment loss for the epoch
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs}")
            optimizer.zero_grad()
            
            # current_alignment_loss_total = 0.0 # Renamed to train_total_alignment_loss

            for batch_idx, batch in enumerate(progress_bar):
                try:
                    # 数据加载和预处理
                    if isinstance(batch, (list, tuple)) and len(batch) == 7: # KGAlignmentDataset
                        images1, input_ids1, attention_mask1, \
                        images2, input_ids2, attention_mask2, labels = batch
                        
                        images1 = images1.to(device)
                        input_ids1 = input_ids1.to(device) if input_ids1 is not None else None
                        attention_mask1 = attention_mask1.to(device) if attention_mask1 is not None else None
                        images2 = images2.to(device)
                        input_ids2 = input_ids2.to(device) if input_ids2 is not None else None
                        attention_mask2 = attention_mask2.to(device) if attention_mask2 is not None else None
                        labels = labels.to(device) # Labels for positive/negative pairs, not used in loss yet

                        outputs = model(images1, input_ids1, attention_mask1,
                                        images2, input_ids2, attention_mask2)

                    elif isinstance(batch, (list, tuple)) and len(batch) == 4:  # TextEnhancedDataset
                        images, input_ids, attention_mask, labels = batch
                        images = images.to(device)
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)
                        # This path will now likely error or behave unexpectedly as model expects 6 inputs
                        # For now, we are focusing on KGAlignment path. This needs to be handled if other datasets are used.
                        print("Warning: TextEnhancedDataset path hit with model expecting paired inputs. Skipping batch or expect error.")
                        # outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
                        continue
                    elif isinstance(batch, (list, tuple)) and len(batch) == 3:  # Flickr8k types
                        images, captions, labels = batch
                        # Similar to above, this path needs adjustment for the new model signature
                        print("Warning: Flickr8kDataset path hit with model expecting paired inputs. Skipping batch or expect error.")
                        continue
                    else:  # Only image input (e.g. raw CIFAR10 without TextEnhancedDataset)
                        # Similar to above, this path needs adjustment
                        print("Warning: Image-only dataset path hit with model expecting paired inputs. Skipping batch or expect error.")
                        continue
                    
                    # 计算损失
                    # logits = outputs['logits'] # Removed: No classification logits
                    router_loss_val = outputs.get('router_loss', torch.tensor(0.0, device=device))
                    router_z_loss = outputs.get('router_z_loss', torch.tensor(0.0, device=device))
                    router_balance_loss = outputs.get('router_balance_loss', torch.tensor(0.0, device=device))
                    cross_modal_loss = outputs.get('cross_modal_loss', torch.tensor(0.0, device=device)) # This is combined from two entities
                    contrastive_loss = outputs.get('contrastive_loss', torch.tensor(0.0, device=device)) # This is combined from two entities

                    # Alignment Loss Calculation
                    embedding1 = outputs['embedding1']
                    embedding2 = outputs['embedding2']

                    embedding1_norm = F.normalize(embedding1, p=2, dim=1)
                    embedding2_norm = F.normalize(embedding2, p=2, dim=1)

                    batch_size_align = embedding1_norm.size(0) # Renamed to avoid conflict with train_loader.batch_size if different

                    # Access margin
                    margin = 0.2 # Default margin
                    if hasattr(train_loader.dataset, 'config') and hasattr(train_loader.dataset.config, 'alignment_margin'):
                        margin = train_loader.dataset.config.alignment_margin

                    sim_matrix = torch.matmul(embedding1_norm, embedding2_norm.t())

                    alignment_loss_val = torch.tensor(0.0, device=device) # Renamed to avoid conflict with alignment_loss in progress bar
                    
                    for i in range(batch_size_align):
                        positive_sim = sim_matrix[i, i]
                        
                        # Term 1: embedding1_i vs embedding2_j (j!=i)
                        negative_sim_e1_vs_e2 = torch.cat((sim_matrix[i, :i], sim_matrix[i, i+1:]))
                        loss_e1_vs_e2 = torch.sum(F.relu(-positive_sim + negative_sim_e1_vs_e2 + margin))
                        alignment_loss_val += loss_e1_vs_e2
                        
                        # Term 2: embedding2_i vs embedding1_j (j!=i) (symmetric part)
                        # sim_matrix_t = sim_matrix.t() # Not needed if we use sim_matrix columns
                        # positive_sim_symmetric = sim_matrix_t[i, i] # same as positive_sim
                        negative_sim_e2_vs_e1 = torch.cat((sim_matrix[:i, i], sim_matrix[i+1:, i]))
                        loss_e2_vs_e1 = torch.sum(F.relu(-positive_sim + negative_sim_e2_vs_e1 + margin))
                        alignment_loss_val += loss_e2_vs_e1
                        
                    if batch_size_align > 0:
                        alignment_loss_val = alignment_loss_val / (2 * batch_size_align)
                    else:
                        alignment_loss_val = torch.tensor(0.0, device=device)

                    train_total_alignment_loss += alignment_loss_val.item()

                    # Total loss
                    loss = alignment_loss_val + router_loss_val
                    
                    # 反向传播和优化
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Accuracy calculation removed
                    # train_total += labels.size(0) # Not relevant for alignment in this way
                    # train_correct += (predicted == labels).sum().item() # Not relevant
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
                    if valid_batches > 0: # train_total removed from condition
                        progress_bar.set_postfix({
                            'loss': f"{train_total_loss/valid_batches:.4f}",
                            # 'acc': f"{100.*train_correct/train_total:.2f}%", # Accuracy removed
                            'align_loss': f"{alignment_loss_val.item():.4f}", # Use the calculated alignment_loss_val
                            'r_loss': f"{router_loss_val.item():.4f}"
                        })
                    
                except Exception as e:
                    print(f"\n处理批次 {batch_idx} 时出错: {str(e)}")
                    if config.debug:
                        import traceback
                        traceback.print_exc()
                    continue
            
            # 计算训练指标
            if valid_batches > 0: # train_total removed from condition
                train_loss = train_total_loss / valid_batches
                # train_acc = 100. * train_correct / train_total # Accuracy removed
                avg_alignment_loss_epoch = train_total_alignment_loss / valid_batches if valid_batches > 0 else 0.0 # Use accumulated
                train_router_z_loss = train_router_z_loss / valid_batches
                train_router_balance_loss = train_router_balance_loss / valid_batches
                train_cross_modal_loss = train_cross_modal_loss / valid_batches
                train_contrastive_loss = train_contrastive_loss / valid_batches
            else:
                print("\n警告：本轮没有有效的训练样本")
                train_loss = float('inf')
                # train_acc = 0.0 # Accuracy removed
                avg_alignment_loss_epoch = 0.0
                train_router_z_loss = 0.0
                train_router_balance_loss = 0.0
                train_cross_modal_loss = 0.0
                train_contrastive_loss = 0.0
            
            # 验证
            model.eval()
            val_loss = 0.0 # Placeholder, evaluate() needs update for alignment
            avg_val_alignment_loss_epoch = 0.0 # Placeholder
            # The evaluate function needs to be refactored for alignment tasks.
            # For now, we'll just get a placeholder loss.
            # test_results = evaluate(model, val_loader, device, criterion=None, class_names=class_descriptions)
            # val_loss = test_results['loss'] # This will be alignment loss + router loss from evaluate
            # val_acc = test_results['accuracy'] # Accuracy not applicable

            # Placeholder validation loop with alignment loss calculation
            temp_val_router_loss_total = 0.0
            temp_val_total_alignment_loss = 0.0 # Accumulator for val alignment loss
            val_batches_count = 0 # Renamed to avoid conflict with valid_batches from training
            with torch.no_grad():
                for val_batch in val_loader:
                    if isinstance(val_batch, (list, tuple)) and len(val_batch) == 7: # KGAlignmentDataset
                        images1, input_ids1, attention_mask1, \
                        images2, input_ids2, attention_mask2, _ = val_batch # labels not used for loss here

                        images1 = images1.to(device)
                        input_ids1 = input_ids1.to(device) if input_ids1 is not None else None
                        attention_mask1 = attention_mask1.to(device) if attention_mask1 is not None else None
                        images2 = images2.to(device)
                        input_ids2 = input_ids2.to(device) if input_ids2 is not None else None
                        attention_mask2 = attention_mask2.to(device) if attention_mask2 is not None else None

                        outputs = model(images1, input_ids1, attention_mask1,
                                        images2, input_ids2, attention_mask2)

                        # Calculate alignment loss for validation batch
                        embedding1_val = outputs['embedding1']
                        embedding2_val = outputs['embedding2']
                        embedding1_norm_val = F.normalize(embedding1_val, p=2, dim=1)
                        embedding2_norm_val = F.normalize(embedding2_val, p=2, dim=1)
                        val_batch_size = embedding1_norm_val.size(0)

                        margin_val = 0.2 # Default
                        if hasattr(val_loader.dataset, 'config') and hasattr(val_loader.dataset.config, 'alignment_margin'):
                            margin_val = val_loader.dataset.config.alignment_margin

                        sim_matrix_val = torch.matmul(embedding1_norm_val, embedding2_norm_val.t())
                        current_val_alignment_loss = torch.tensor(0.0, device=device)
                        for i in range(val_batch_size):
                            pos_sim = sim_matrix_val[i,i]
                            neg_sim_e1_e2 = torch.cat((sim_matrix_val[i, :i], sim_matrix_val[i, i+1:]))
                            current_val_alignment_loss += torch.sum(F.relu(-pos_sim + neg_sim_e1_e2 + margin_val))
                            neg_sim_e2_e1 = torch.cat((sim_matrix_val[:i, i], sim_matrix_val[i+1:, i]))
                            current_val_alignment_loss += torch.sum(F.relu(-pos_sim + neg_sim_e2_e1 + margin_val))

                        if val_batch_size > 0:
                             current_val_alignment_loss = current_val_alignment_loss / (2 * val_batch_size)
                        else:
                             current_val_alignment_loss = torch.tensor(0.0, device=device)

                        temp_val_total_alignment_loss += current_val_alignment_loss.item()
                        router_loss_val_batch = outputs.get('router_loss', torch.tensor(0.0, device=device))
                        temp_val_router_loss_total += router_loss_val_batch.item()
                        val_batches_count +=1
                    else:
                        pass
            if val_batches_count > 0:
                val_loss = (temp_val_total_alignment_loss + temp_val_router_loss_total) / val_batches_count
                avg_val_alignment_loss_epoch = temp_val_total_alignment_loss / val_batches_count
            else:
                print("Warning: No valid batches in validation loader for KGAlignment type.")
                val_loss = float('inf')
                avg_val_alignment_loss_epoch = 0.0

            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # 保存指标
            metrics['train_loss'].append(train_loss)
            # metrics['train_acc'].append(train_acc) # Accuracy removed
            metrics['train_alignment_loss'].append(avg_alignment_loss_epoch)
            metrics['val_loss'].append(val_loss)
            # metrics['val_acc'].append(val_acc) # Accuracy removed
            metrics['val_alignment_loss'].append(avg_val_alignment_loss_epoch)
            metrics['lr'].append(current_lr)
            metrics['router_z_loss'].append(train_router_z_loss)
            metrics['router_balance_loss'].append(train_router_balance_loss)
            metrics['cross_modal_loss'].append(train_cross_modal_loss)
            metrics['contrastive_loss'].append(train_contrastive_loss)
            
            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{config.training.num_epochs} 完成 ({epoch_time:.2f}s)")
            # print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%") # Accuracy removed
            print(f"训练损失: {train_loss:.4f}, 训练对齐损失: {avg_alignment_loss_epoch:.4f}")
            # print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%") # Accuracy removed
            print(f"验证损失: {val_loss:.4f}, 验证对齐损失: {avg_val_alignment_loss_epoch:.4f}")
            print(f"路由损失: Z={train_router_z_loss:.6f}, Balance={train_router_balance_loss:.6f}, CrossModal={train_cross_modal_loss:.6f}, Contrastive={train_contrastive_loss:.6f}")
            print(f"学习率: {current_lr:.8f}")
            
            # 保存最佳模型 - based on val_loss now (lower is better)
            # Initialize best_val_loss appropriately, e.g., float('inf')
            if 'best_val_loss' not in locals(): best_val_loss = float('inf')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss, # Changed from best_val_acc
                    'metrics': metrics
                }, save_path)
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
    
    except Exception as e:
        print(f"\n训练过程中出错: {str(e)}")
        if config.debug:
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n训练完成，总用时: {total_time:.2f}s")
    # print(f"最佳验证准确率: {best_val_acc:.2f}%") # Accuracy removed
    if 'best_val_loss' in locals():
        print(f"最佳验证损失: {best_val_loss:.4f}")
    else:
        print("未能确定最佳验证损失。")

    return model, metrics