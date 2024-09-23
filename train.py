import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import InfoNCELoss
from cross_attention import CrossAttention
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    info_nce_loss = InfoNCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    cross_attn = CrossAttention().to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (original_images, input_ids, attention_mask, labels) in enumerate(dataloader):
            original_images = original_images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            classification_output, image_feature_vector, text_feature_vector = model(original_images, input_ids, attention_mask)

            contrastive_loss = info_nce_loss(image_feature_vector, text_feature_vector)
            classification_loss = criterion(classification_output, labels)
            loss = contrastive_loss + classification_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_contrastive_loss = contrastive_loss.item() / len(dataloader)
        avg_classification_loss = classification_loss.item() / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, "
              f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
              f"Classification Loss: {avg_classification_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'model_version': '1.2'
    }, save_path)

    print("训练完成。")
