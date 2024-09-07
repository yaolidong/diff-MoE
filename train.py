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

            classification_output, image_first_vector, image_second_vector, image_cls, \
            text_first_vector, text_second_vector, text_cls= model(original_images, input_ids, attention_mask)
            loss_image = contrastive_loss(image_first_vector, image_second_vector)
            loss_text = contrastive_loss(text_first_vector, text_second_vector)
            loss_text_image = contrastive_loss(image_second_vector, text_second_vector)
            loss_image_cls = F.cross_entropy(image_cls, labels)
            loss_text_cls = F.cross_entropy(text_cls, labels)
            loss = loss_image_cls + loss_text_cls + 0.5 * loss_text_image + 0.5 * loss_text + 0.5 * loss_image  # 可以调整对比损失的权重

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'model_version': '1.2'
    }, save_path)

    print("Training completed.")
