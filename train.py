import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import contrastive_loss, compute_loss  # 导入新的loss函数

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (original_images, augmented_images, input_ids, attention_mask, labels) in enumerate(dataloader):
            original_images = original_images.to(device)
            augmented_images = augmented_images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # 原始图像前向传播
            orig_image_first, orig_image_second, orig_image_cls_first, orig_image_cls_second, \
            text_first, text_second, text_cls_first, text_cls_second, orig_outputs = model(original_images, input_ids, attention_mask)

            # 增强图像前向传播
            aug_image_first, aug_image_second, aug_image_cls_first, aug_image_cls_second, \
            _, _, _, _, aug_outputs = model(augmented_images, input_ids, attention_mask)
            print(f"orig_image_second的维度: {orig_image_second.shape}")
            print(f"aug_image_second的维度: {aug_image_second.shape}")
            # 计算对比损失
            loss_image = contrastive_loss(orig_image_second, aug_image_second)
            # loss_text_image = contrastive_loss(text_second, orig_image_second)

            # 计算分类损失
            loss_cls = F.cross_entropy(orig_outputs, labels) + F.cross_entropy(aug_outputs, labels)

            # 总损失
            loss = loss_image + loss_cls

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training completed.")
