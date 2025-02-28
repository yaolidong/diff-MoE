import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
import numpy as np
from config import TrainingConfig
from transformers import get_cosine_schedule_with_warmup
import os
from typing import Tuple, Dict, List
# 添加混合精度训练支持
from torch.amp import autocast, GradScaler

# 配置日志 - 设置为WARNING级别，减少INFO消息
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 只为训练模块保留INFO级别

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
               gradient_clip_val=1.0, use_amp=True, grad_accum_steps=1, text_tokens=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_router_loss = 0
    total_samples = 0
    correct = 0
    
    # 创建梯度缩放器（用于混合精度训练）
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    # 重置梯度
    optimizer.zero_grad()
    
    # 跟踪NaN出现次数
    nan_count = 0
    max_nan_tolerance = 10
    
    # 跟踪错误统计
    error_types = {}
    total_batches = len(train_loader)
    processed_batches = 0
    skipped_batches = 0
    
    logger.info(f"开始训练Epoch {epoch}, 总批次数: {total_batches}")
    if use_amp:
        logger.info(f"使用混合精度训练, 梯度累积步数: {grad_accum_steps}")
    
    # 记录是否使用文本模态
    if text_tokens is not None:
        logger.info(f"使用文本模态进行训练，文本令牌形状: {text_tokens.shape}")
    
    # 设置进度条，减少刷新频率以降低CPU开销
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', 
                        miniters=max(1, len(train_loader)//20),
                        dynamic_ncols=True)
    for batch_idx, batch_data in enumerate(progress_bar):
        try:
            # 确保数据格式正确
            if isinstance(batch_data, (tuple, list)):
                if len(batch_data) == 2:
                    data, target = batch_data
                else:
                    error_msg = f"Expected batch data length 2, got {len(batch_data)}"
                    logger.error(f"批次 {batch_idx} 格式错误: {error_msg}")
                    skipped_batches += 1
                    error_types.setdefault("数据格式错误", 0)
                    error_types["数据格式错误"] += 1
                    raise ValueError(error_msg)
            else:
                error_msg = f"Expected tuple or list, got {type(batch_data)}"
                logger.error(f"批次 {batch_idx} 类型错误: {error_msg}")
                skipped_batches += 1
                error_types.setdefault("数据类型错误", 0)
                error_types["数据类型错误"] += 1
                raise ValueError(error_msg)
            
            # 移动数据到设备
            data, target = data.to(device), target.to(device)
            
            # 记录简要的数据统计
            if batch_idx == 0:
                logger.info(f"首批数据形状: {data.shape}, 数据类型: {data.dtype}, 值范围: {data.min().item()} 到 {data.max().item()}")
            
            # 使用自动混合精度
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    # 前向传播，如果有文本令牌则传入
                    if text_tokens is not None:
                        outputs = model(data, text_tokens=text_tokens)
                    else:
                        outputs = model(data)
                    
                    logits = outputs['logits']
                    router_loss = outputs['router_loss']
                    
                    # 检查数据类型
                    if batch_idx == 0:
                        logger.info(f"logits dtype: {logits.dtype}, router_loss dtype: {router_loss.dtype}")
                    
                    # 计算损失
                    cls_loss = criterion(logits, target)
                    loss = cls_loss + TrainingConfig.router_loss_weight * router_loss
                    # 缩放损失以进行梯度累积
                    loss = loss / grad_accum_steps
                
                # 反向传播（使用梯度缩放）
                scaler.scale(loss).backward()
                
                # 梯度累积
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪
                    if gradient_clip_val > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                    
                    # 检查梯度是否出现NaN或Inf
                    has_nan_grad = False
                    has_inf_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                            if torch.isinf(param.grad).any():
                                has_inf_grad = True
                                break
                    
                    # 只在梯度正常时更新参数
                    if not has_nan_grad and not has_inf_grad:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logger.warning(f"批次 {batch_idx} 检测到NaN/Inf梯度，跳过参数更新")
                    
                    optimizer.zero_grad()
            else:
                # 前向传播 (不使用混合精度)
                if text_tokens is not None:
                    outputs = model(data, text_tokens=text_tokens)
                else:
                    outputs = model(data)
                
                logits = outputs['logits']
                router_loss = outputs['router_loss']
                
                # 计算损失
                cls_loss = criterion(logits, target)
                loss = cls_loss + TrainingConfig.router_loss_weight * router_loss
                # 缩放损失以进行梯度累积
                loss = loss / grad_accum_steps
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪
                    if gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                        # 添加参数值裁剪，放宽限制
                        for p in model.parameters():
                            p.data.clamp_(-20, 20)  # 放宽参数值范围限制
                    
                    # 检查梯度是否出现NaN或Inf
                    has_nan_grad = False
                    has_inf_grad = False
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                has_nan_grad = True
                                break
                            if torch.isinf(param.grad).any():
                                has_inf_grad = True
                                break
                    
                    # 只在梯度正常时更新参数
                    if not has_nan_grad and not has_inf_grad:
                        optimizer.step()
                    else:
                        logger.warning(f"批次 {batch_idx} 检测到NaN/Inf梯度，跳过参数更新")
                    
                    optimizer.zero_grad()
            
            # 检测NaN
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"批次 {batch_idx} 检测到NaN/Inf损失，跳过当前批次")
                skipped_batches += 1
                error_types.setdefault("NaN/Inf损失", 0)
                error_types["NaN/Inf损失"] += 1
                
                # 提供更详细的诊断信息
                logger.debug(f"数据范围: {data.min().item()} to {data.max().item()}")
                logger.debug(f"logits范围: {logits.min().item()} to {logits.max().item()}")
                logger.debug(f"数据类型 - data: {data.dtype}, logits: {logits.dtype}, loss: {loss.dtype}")
                
                # 在第一批次输出更多调试信息
                if batch_idx == 0:
                    logger.info("第一批次数据类型信息:")
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.info(f"{name}: {param.dtype}")
                
                continue
            
            # 统计（使用原始损失值，不考虑梯度累积缩放）
            actual_loss = loss.item() * grad_accum_steps
            batch_size = data.size(0)
            total_loss += actual_loss * batch_size
            total_router_loss += router_loss.item() * batch_size
            total_samples += batch_size
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            processed_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                    'loss': f'{actual_loss:.4f}',
                    'router_loss': f'{router_loss.item():.4f}',
                    'acc': f'{100. * correct / total_samples:.2f}%'
                })
            
        except Exception as e:
            logger.error(f"训练批次 {batch_idx} 出错: {str(e)}")
            skipped_batches += 1
            error_type = type(e).__name__
            error_types.setdefault(error_type, 0)
            error_types[error_type] += 1
            continue
    
    # 输出详细的错误统计
    logger.info(f"Epoch {epoch} 统计: 总批次数={total_batches}, 处理批次数={processed_batches}, 跳过批次数={skipped_batches}")
    if skipped_batches > 0:
        logger.warning(f"错误类型统计: {error_types}")
    
    # 计算平均损失和准确率
    if total_samples == 0:
        logger.error(f"训练过程中没有处理任何样本! 批次总数: {len(train_loader)}, 成功处理批次: 0")
        # 返回占位符结果，避免后续处理崩溃
        placeholder_results = {
            'loss': 0.0,
            'router_loss': 0.0,
            'accuracy': 0.0,
            'total_loss': 0.0,
            'error': f"没有处理任何样本，所有批次都出错。错误统计: {error_types}"
        }
        return placeholder_results
    
    avg_loss = total_loss / total_samples
    avg_router_loss = total_router_loss / total_samples
    accuracy = 100. * correct / total_samples
    
    # 构建返回字典
    results = {
        'loss': avg_loss,
        'router_loss': avg_router_loss,
        'accuracy': accuracy,
        'total_loss': avg_loss + TrainingConfig.router_loss_weight * avg_router_loss
    }
    
    # 输出统计信息，帮助调试
    logger.info(f"训练epoch完成: 总样本数={total_samples}, 总批次数={len(train_loader)}")
    
    return results

def validate(model, val_loader, criterion, device, text_tokens=None):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    total_router_loss = 0
    total_samples = 0
    correct = 0
    
    # 跟踪错误统计
    error_types = {}
    total_batches = len(val_loader)
    processed_batches = 0
    skipped_batches = 0
    
    logger.info(f"开始验证, 总批次数: {total_batches}")
    
    # 记录是否使用文本模态
    if text_tokens is not None:
        logger.info(f"使用文本模态进行验证，文本令牌形状: {text_tokens.shape}")
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validating')
        for batch_idx, batch_data in enumerate(progress_bar):
            try:
                # 确保数据格式正确
                if isinstance(batch_data, (tuple, list)):
                    if len(batch_data) == 2:
                        data, target = batch_data
                    else:
                        error_msg = f"Expected batch data length 2, got {len(batch_data)}"
                        logger.error(f"验证批次 {batch_idx} 格式错误: {error_msg}")
                        skipped_batches += 1
                        error_types.setdefault("数据格式错误", 0)
                        error_types["数据格式错误"] += 1
                        raise ValueError(error_msg)
                else:
                    error_msg = f"Expected tuple or list, got {type(batch_data)}"
                    logger.error(f"验证批次 {batch_idx} 类型错误: {error_msg}")
                    skipped_batches += 1
                    error_types.setdefault("数据类型错误", 0)
                    error_types["数据类型错误"] += 1
                    raise ValueError(error_msg)
                
                # 记录首批数据统计
                if batch_idx == 0:
                    logger.info(f"验证首批数据形状: {data.shape}, 类型: {data.dtype}, 值范围: {data.min().item()} 到 {data.max().item()}")
                
                data, target = data.to(device), target.to(device)
                
                # 前向传播，如果有文本令牌则传入
                if text_tokens is not None:
                    outputs = model(data, text_tokens=text_tokens)
                else:
                    outputs = model(data)
                
                logits = outputs['logits']
                router_loss = outputs['router_loss']
                
                # 计算损失
                cls_loss = criterion(logits, target)
                loss = cls_loss + TrainingConfig.router_loss_weight * router_loss
                
                # 检测NaN/Inf
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(f"验证批次 {batch_idx} 检测到NaN/Inf损失，跳过")
                    skipped_batches += 1
                    error_types.setdefault("NaN/Inf损失", 0)
                    error_types["NaN/Inf损失"] += 1
                    continue
                
                # 统计
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_router_loss += router_loss.item() * batch_size
                total_samples += batch_size
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                processed_batches += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'router_loss': f'{router_loss.item():.4f}',
                    'acc': f'{100. * correct / total_samples:.2f}%'
                })
                
            except Exception as e:
                logger.error(f"验证批次 {batch_idx} 出错: {str(e)}")
                skipped_batches += 1
                error_type = type(e).__name__
                error_types.setdefault(error_type, 0)
                error_types[error_type] += 1
                continue
    
    # 输出详细的错误统计
    logger.info(f"验证统计: 总批次数={total_batches}, 处理批次数={processed_batches}, 跳过批次数={skipped_batches}")
    if skipped_batches > 0:
        logger.warning(f"验证错误类型统计: {error_types}")
    
    # 计算平均损失和准确率
    if total_samples == 0:
        logger.error(f"验证过程中没有处理任何样本! 批次总数: {len(val_loader)}, 成功处理批次: 0")
        # 返回占位符结果，避免后续处理崩溃
        placeholder_results = {
            'loss': 0.0,
            'router_loss': 0.0,
            'accuracy': 0.0,
            'total_loss': 0.0,
            'error': f"验证没有处理任何样本，所有批次都出错。错误统计: {error_types}"
        }
        return placeholder_results
    
    avg_loss = total_loss / total_samples
    avg_router_loss = total_router_loss / total_samples
    accuracy = 100. * correct / total_samples
    
    # 构建返回字典
    results = {
        'loss': avg_loss,
        'router_loss': avg_router_loss,
        'accuracy': accuracy,
        'total_loss': avg_loss + TrainingConfig.router_loss_weight * avg_router_loss
    }
    
    # 输出统计信息，帮助调试
    logger.info(f"验证完成: 总样本数={total_samples}, 总批次数={len(val_loader)}")
    
    return results

def create_optimizer(model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
    """根据配置创建优化器"""
    return optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

def create_scheduler(optimizer, train_loader, config):
    # 计算总的训练步数
    num_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    return CosineAnnealingWarmRestarts(
        optimizer,
        T_0=num_training_steps,
        T_mult=1,
        eta_min=0
    )

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    num_epochs: int = TrainingConfig.num_epochs,
    learning_rate: float = TrainingConfig.learning_rate,
    weight_decay: float = TrainingConfig.weight_decay,
    gradient_clip_val: float = TrainingConfig.gradient_clip_val,
    early_stopping_patience: int = TrainingConfig.early_stopping_patience,
    warmup_epochs: int = TrainingConfig.warmup_epochs,
    label_smoothing: float = TrainingConfig.label_smoothing,
    use_amp: bool = True,
    grad_accum_steps: int = 1,
    criterion: nn.Module = None,
    optimizer: optim.Optimizer = None,
    scheduler = None,
    class_descriptions = None  # 新增参数：类别文本描述
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 运行设备
        save_path: 模型保存路径
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        gradient_clip_val: 梯度裁剪值
        early_stopping_patience: 早停耐心值
        warmup_epochs: 预热轮数
        label_smoothing: 标签平滑值
        use_amp: 是否使用混合精度训练
        grad_accum_steps: 梯度累积步数
        criterion: 损失函数，如果为None则创建
        optimizer: 优化器，如果为None则创建
        scheduler: 学习率调度器，如果为None则创建
        class_descriptions: 类别文本描述字典，用于多模态训练
        
    Returns:
        训练好的模型和训练历史
    """
    logger.info("开始训练...")
    
    # 确保保存目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 如果未提供，创建优化器
    if optimizer is None:
        optimizer = create_optimizer(model, TrainingConfig)
    
    # 如果未提供，创建学习率调度器
    if scheduler is None:
        scheduler = create_scheduler(optimizer, train_loader, TrainingConfig)
    
    # 如果未提供，创建损失函数
    if criterion is None:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 准备文本令牌（如果有文本描述）
    text_tokens = None
    if class_descriptions is not None:
        # 这里简化处理，实际应用中应该使用tokenizer
        # 将文本描述转换为简单的数字序列作为示例
        text_tokens = torch.randint(0, 1000, (len(class_descriptions), 32), device=device)
        logger.info(f"已准备{len(class_descriptions)}个类别的文本描述令牌")
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_router_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_router_loss': [],
        'val_accuracy': []
    }
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    try:
        for epoch in range(num_epochs):
            # 训练一轮
            train_results = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                gradient_clip_val=gradient_clip_val,
                use_amp=use_amp,  # 传递混合精度参数
                grad_accum_steps=grad_accum_steps,  # 传递梯度累积参数
                text_tokens=text_tokens  # 传递文本令牌
            )
            
            # 检查训练结果是否包含错误
            if 'error' in train_results:
                logger.error(f"训练过程出错: {train_results['error']}")
                # 尝试使用安全值继续训练
                logger.info("使用默认值继续训练流程...")
            
            # 验证
            val_results = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                text_tokens=text_tokens  # 传递文本令牌
            )
            
            # 检查验证结果是否包含错误
            if 'error' in val_results:
                logger.error(f"验证过程出错: {val_results['error']}")
                logger.info("使用默认值继续训练流程...")
            
            # 更新学习率
            scheduler.step()
            
            # 更新历史
            history['train_loss'].append(train_results['loss'])
            history['train_router_loss'].append(train_results['router_loss'])
            history['train_accuracy'].append(train_results['accuracy'])
            history['val_loss'].append(val_results['loss'])
            history['val_router_loss'].append(val_results['router_loss'])
            history['val_accuracy'].append(val_results['accuracy'])
            
            # 打印进度
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_results['total_loss']:.4f} - "
                f"Train Acc: {train_results['accuracy']:.2f}% - "
                f"Val Loss: {val_results['total_loss']:.4f} - "
                f"Val Acc: {val_results['accuracy']:.2f}%"
            )
        
            # 早停检查
            val_loss = val_results['total_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                
                # 保存最佳模型
                try:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'history': history,
                        'best_val_loss': best_val_loss
                    }
                    torch.save(checkpoint, str(save_path))
                    logger.info(f"保存最佳模型到 {save_path}")
                except Exception as e:
                    logger.error(f"保存模型时出错: {str(e)}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停：验证损失在 {early_stopping_patience} 轮内没有改善")
                    break
    
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """获取带预热的余弦学习率调度器"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 