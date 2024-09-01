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
        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(dataloader):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = F.cross_entropy(outputs, labels)
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
