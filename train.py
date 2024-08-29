import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(x, y):
    # 假设x和y是已经归一化的向量
    similarity = torch.matmul(x, y.t())
    # 对角线上的元素应该最大化,其他元素应该最小化
    labels = torch.arange(x.size(0)).to(x.device)
    return F.cross_entropy(similarity, labels)

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, texts, labels) in enumerate(dataloader):
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            img_first_vec, img_second_vec, img_cls_first, img_cls_second, txt_first_vec, txt_second_vec, txt_cls_first, txt_cls_second = model(images, texts)

            # 计算损失
            loss_cls1 = contrastive_loss(img_cls_first, txt_cls_first)
            loss_cls2 = contrastive_loss(img_cls_second, txt_cls_second)
            loss_img = contrastive_loss(img_first_vec, img_second_vec)
            loss_txt = contrastive_loss(txt_first_vec, txt_second_vec)
            
            loss = loss_cls1 + loss_cls2 + loss_img + loss_txt
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Average loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    return model
