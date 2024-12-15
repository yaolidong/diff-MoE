import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from losses import InfoNCELoss
from cross_attention import CrossAttention

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class WarmupLinearScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / self.warmup_steps) * self.optimizer.param_groups[0]['lr']
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = max(self.min_lr, self.optimizer.param_groups[0]['lr'] * (1 - progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train(model, train_loader, val_loader, optimizer, device, num_epochs, save_path, 
          patience=7, warmup_epochs=5):
    criterion = nn.CrossEntropyLoss().to(device)
    
    # 初始化学习率调度器
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    warmup_scheduler = WarmupLinearScheduler(optimizer, warmup_steps, total_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=patience)
    
    # 用于记录每个 epoch 的损失
    epoch_losses = {
        'total': [], 'contrastive': [], 'image_cls': [], 
        'text_cls': [], 'fused_cls': [], 'reconstruction': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        contrastive_loss_sum = 0
        image_cls_loss_sum = 0
        text_cls_loss_sum = 0
        fused_cls_loss_sum = 0
        reconstruction_loss_sum = 0
        
        dataloader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(dataloader_tqdm):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images, input_ids, attention_mask)
            if isinstance(outputs, tuple):
                fused_cls = outputs[4]  # 获取分类输出
            else:
                fused_cls = outputs
            
            # 确保维度正确
            if len(fused_cls.shape) == 3:
                fused_cls = fused_cls.squeeze(1)
            
            # 确保 labels 是一维的
            if len(labels.shape) > 1:
                labels = labels.squeeze()
            
            # 计算损失
            try:
                loss = criterion(fused_cls, labels)
            except Exception as e:
                print(f"Error in loss calculation:")
                print(f"fused_cls shape: {fused_cls.shape}")
                print(f"labels shape: {labels.shape}")
                raise e
            
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            if epoch < warmup_epochs:
                current_lr = warmup_scheduler.step()
            else:
                cosine_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            
            # 累计损失
            total_loss += loss.item()
            contrastive_loss_sum += 0  # 这里没有对比损失，所以保持为0
            image_cls_loss_sum += 0  # 这里没有图像分类损失，所以保持为0
            text_cls_loss_sum += 0  # 这里没有文本分类损失，所以保持为0
            fused_cls_loss_sum += 0  # 这里没有融合分类损失，所以保持为0
            reconstruction_loss_sum += 0  # 这里没有重建损失，所以保持为0
            
            # 更新进度条
            dataloader_tqdm.set_postfix({
                'Loss': f"{loss.item():.2f}",
                'LR': f"{current_lr:.2e}",
                'Contrastive Loss': f"{0:.2f}"
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_contrastive_loss = contrastive_loss_sum / len(train_loader)
        avg_image_cls_loss = image_cls_loss_sum / len(train_loader)
        avg_text_cls_loss = text_cls_loss_sum / len(train_loader)
        avg_fused_cls_loss = fused_cls_loss_sum / len(train_loader)
        avg_reconstruction_loss = reconstruction_loss_sum / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                outputs = model(images, input_ids, attention_mask)
                if isinstance(outputs, tuple):
                    fused_cls = outputs[4]
                else:
                    fused_cls = outputs
                
                val_loss += criterion(fused_cls, labels).item()
        
        val_loss /= len(val_loader)
        
        # 记录损失
        epoch_losses['total'].append(avg_loss)
        epoch_losses['contrastive'].append(avg_contrastive_loss)
        epoch_losses['image_cls'].append(avg_image_cls_loss)
        epoch_losses['text_cls'].append(avg_text_cls_loss)
        epoch_losses['fused_cls'].append(avg_fused_cls_loss)
        epoch_losses['reconstruction'].append(avg_reconstruction_loss)
        epoch_losses['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("保存最佳模型")
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("触发早停！")
            break
    
    # 绘制损失曲线
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 1, 1)
    epochs = range(1, len(epoch_losses['total']) + 1)
    plt.plot(epochs, epoch_losses['total'], label='Total Loss')
    plt.plot(epochs, epoch_losses['contrastive'], label='Contrastive Loss')
    plt.plot(epochs, epoch_losses['image_cls'], label='Image Classification Loss')
    plt.plot(epochs, epoch_losses['text_cls'], label='Text Classification Loss')
    plt.plot(epochs, epoch_losses['fused_cls'], label='Fused Classification Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 学习率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epochs, epoch_losses['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    print("训练曲线已保存为 'training_curves.png'") 