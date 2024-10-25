import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from losses import InfoNCELoss
from cross_attention import CrossAttention

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    info_nce_loss = InfoNCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    cross_attn = CrossAttention().to(device)
    
    # 用于记录每个 epoch 的损失
    epoch_losses = {
        'total': [],
        'contrastive': [],
        'image_cls': [],
        'text_cls': [],
        'fused_cls': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        contrastive_loss_sum = 0
        image_cls_loss_sum = 0
        text_cls_loss_sum = 0
        fused_cls_loss_sum = 0
        
        dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(dataloader_tqdm):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            image_feature_vector, text_feature_vector, image_cls, text_cls, fused_cls, (first_expert_outputs, second_expert_outputs), (first_gating_output, second_gating_output) = model(images, input_ids, attention_mask)
            contrastive_loss = info_nce_loss(image_feature_vector, text_feature_vector)
            image_cls_loss = criterion(image_cls, labels)
            text_cls_loss = criterion(text_cls, labels)
            fused_cls_loss = criterion(fused_cls, labels)
            
            # 调整损失权重
            loss = 0.2 * contrastive_loss + 0.2 * image_cls_loss + 0.2 * text_cls_loss + 0.4 * fused_cls_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            image_cls_loss_sum += image_cls_loss.item()
            text_cls_loss_sum += text_cls_loss.item()
            fused_cls_loss_sum += fused_cls_loss.item()

            # 更新进度条描述
            dataloader_tqdm.set_postfix({
                'Loss': loss.item(),
                'Contrastive Loss': contrastive_loss.item(),
                'Image Classification Loss': image_cls_loss.item(),
                'Text Classification Loss': text_cls_loss.item(),
                'Fused Classification Loss': fused_cls_loss.item()
            })

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_contrastive_loss = contrastive_loss_sum / len(dataloader)
        avg_image_cls_loss = image_cls_loss_sum / len(dataloader)
        avg_text_cls_loss = text_cls_loss_sum / len(dataloader)
        avg_fused_cls_loss = fused_cls_loss_sum / len(dataloader)

        # 记录损失
        epoch_losses['total'].append(avg_loss)
        epoch_losses['contrastive'].append(avg_contrastive_loss)
        epoch_losses['image_cls'].append(avg_image_cls_loss)
        epoch_losses['text_cls'].append(avg_text_cls_loss)
        epoch_losses['fused_cls'].append(avg_fused_cls_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print("训练完成。模型已保存。")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, epoch_losses['total'], label='Total Loss')
    plt.plot(epochs, epoch_losses['contrastive'], label='Contrastive Loss')
    plt.plot(epochs, epoch_losses['image_cls'], label='Image Classification Loss')
    plt.plot(epochs, epoch_losses['text_cls'], label='Text Classification Loss')
    plt.plot(epochs, epoch_losses['fused_cls'], label='Fused Classification Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    print("损失曲线已保存为 'loss_curve.png'")
