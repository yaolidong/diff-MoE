import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
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

def train(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cuda',
          early_stopping_patience=10):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_router_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证
        val_loss, val_router_loss, val_acc = validate(
            model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印统计
        print(f'Epoch {epoch}:')
        print(f'Train - Loss: {train_loss:.4f}, Router Loss: {train_router_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Val - Loss: {val_loss:.4f}, Router Loss: {val_router_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    return model 