import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import contrastive_loss
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (original_images, input_ids, attention_mask, labels) in enumerate(dataloader):
            original_images = original_images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            classification_output, image_first_vector, image_second_vector, image_vector, image_cls, \
            text_first_vector, text_second_vector, text_vector, text_cls= model(original_images, input_ids, attention_mask)

            loss_text_image = contrastive_loss(image_vector, text_vector)
            loss_cls = F.cross_entropy(classification_output, labels)
            loss = loss_cls + 0.5 * loss_text_image  # 可以调整对比损失的权重

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Total Loss: {loss.item():.4f}, Cls Loss: {loss_cls.item():.4f}, Contrastive Loss: {loss_text_image.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'model_version': '1.2'
    }, save_path)

    print("Training completed.")
