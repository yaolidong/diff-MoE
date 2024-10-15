import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # 导入tqdm

from losses import InfoNCELoss
from cross_attention import CrossAttention

def train(model, dataloader, optimizer, device, num_epochs, save_path):
    info_nce_loss = InfoNCELoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    cross_attn = CrossAttention().to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        dataloader_tqdm = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(dataloader_tqdm):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            image_feature_vector, text_feature_vector, image_cls, text_cls = model(images, input_ids, attention_mask)

            contrastive_loss = info_nce_loss(image_feature_vector, text_feature_vector)
            image_cls_loss = criterion(image_cls, labels)
            text_cls_loss = criterion(text_cls, labels)
            loss =  contrastive_loss + image_cls_loss + text_cls_loss 

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条描述
            dataloader_tqdm.set_postfix({
                'Loss': loss.item(),
                'Contrastive Loss': contrastive_loss.item(),
                'Image Classification Loss': image_cls_loss.item(),
                'Text Classification Loss': text_cls_loss.item()
            })

        avg_loss = total_loss / len(dataloader)
        avg_contrastive_loss = contrastive_loss.item() / len(dataloader)
        avg_image_cls_loss = image_cls_loss.item() / len(dataloader)
        avg_text_cls_loss = text_cls_loss.item() / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, "
              f"Contrastive Loss: {avg_contrastive_loss:.4f}, "
              f"Image Classification Loss: {avg_image_cls_loss:.4f}, "
              f"Text Classification Loss: {avg_text_cls_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'model_version': '1.2'
    }, save_path)

    print("训练完成。")
