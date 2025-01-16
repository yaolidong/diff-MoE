import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, gradient_clip_val=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_router_loss = 0
    total_samples = 0
    correct = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        logits = outputs['logits']
        router_loss = outputs['router_loss']
        
        # 计算损失
        cls_loss = criterion(logits, target)
        loss = cls_loss + 0.01 * router_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        
        optimizer.step()
        
        # 统计
        batch_size = data.size(0)
        total_loss += cls_loss.item() * batch_size
        total_router_loss += router_loss.item() * batch_size
        total_samples += batch_size
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': total_loss / total_samples,
            'router_loss': total_router_loss / total_samples,
            'acc': 100. * correct / total_samples
        })
    
    return total_loss / total_samples, total_router_loss / total_samples, 100. * correct / total_samples

def validate(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    total_router_loss = 0
    total_samples = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            outputs = model(data)
            logits = outputs['logits']
            router_loss = outputs['router_loss']
            
            # 计算损失
            cls_loss = criterion(logits, target)
            
            # 统计
            batch_size = data.size(0)
            total_loss += cls_loss.item() * batch_size
            total_router_loss += router_loss.item() * batch_size
            total_samples += batch_size
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    return total_loss / total_samples, total_router_loss / total_samples, 100. * correct / total_samples

def train(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda',
          early_stopping_patience=10, warmup_epochs=5, weight_decay=0.01,
          gradient_clip_val=1.0, label_smoothing=0.1, checkpoint_path=None):
    """
    训练模型的函数
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        early_stopping_patience: 早停耐心值
        warmup_epochs: 预热轮数
        weight_decay: 权重衰减
        gradient_clip_val: 梯度裁剪值
        label_smoothing: 标签平滑值
        checkpoint_path: 检查点保存路径
    Returns:
        训练好的模型
    """
    # 确保model目录存在
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # 学习率调度器
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 早停
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 如果存在检查点，加载它
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        history = checkpoint['history']
        logger.info(f"从轮次 {start_epoch} 继续训练")
    
    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_loss, train_router_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, gradient_clip_val)
        
        # 验证
        val_loss, val_router_loss, val_acc = validate(
            model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印统计
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train - Loss: {train_loss:.4f}, Router Loss: {train_router_loss:.4f}, Acc: {train_acc:.2f}%')
        logger.info(f'Val - Loss: {val_loss:.4f}, Router Loss: {val_router_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 保存检查点
        if checkpoint_path:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            # 保存最佳模型
            best_model_path = checkpoint_path.replace('checkpoint_', 'best_model_')
            torch.save(best_model_state, best_model_path)
            logger.info(f"保存最佳模型到 {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停：验证损失在 {early_stopping_patience} 轮内没有改善")
                break
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info("已恢复到最佳模型状态")
    
    return model 