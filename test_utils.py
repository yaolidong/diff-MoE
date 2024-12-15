import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import label_to_text
import matplotlib
import matplotlib.font_manager as fm
import numpy as np

matplotlib.use('Agg')

def calculate_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def test(model, dataloader, device, label_to_text_map):
    model.eval()
    all_preds = []
    all_labels = []
    reconstruction_loss = 0
    
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # 获取模型输出
            (image_feature_vector, text_feature_vector, 
             image_cls, text_cls, fused_cls,
             text_reconstruction,
             expert_outputs, gating_outputs) = model(images, input_ids, attention_mask)
            
            # 使用融合后的分类结果
            _, preds = torch.max(fused_cls, 1)
            
            # 计算文本重建损失
            recon_loss = nn.CrossEntropyLoss(ignore_index=0)(
                text_reconstruction.view(-1, text_reconstruction.size(-1)),
                input_ids.view(-1)
            )
            reconstruction_loss += recon_loss.item()
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    avg_reconstruction_loss = reconstruction_loss / len(dataloader)
    
    print("\n预测结果:")
    print_output_distribution(fused_cls)
    for pred, label in zip(all_preds[:10], all_labels[:10]):  # 只打印前10个结果
        pred_text = label_to_text_map[pred]
        true_text = label_to_text_map[label]
        print(f"预测: {pred_text}, 实际: {true_text}")
    
    print(f"\n准确率: {accuracy:.4f}")
    print(f"平均重建损失: {avg_reconstruction_loss:.4f}")
    
    return accuracy, avg_reconstruction_loss

def visualize_predictions(model, dataloader, device, label_to_text_map):
    model.eval()
    images, input_ids, attention_mask, labels = next(iter(dataloader))
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    with torch.no_grad():
        image_feature_vector, text_feature_vector, image_cls, text_cls, fused_cls, text_reconstruction, (first_expert_outputs, second_expert_outputs), (first_gating_output, second_gating_output) = model(images, input_ids, attention_mask)
        _, preds = torch.max(image_cls, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].cpu().numpy()
            if img.shape[0] == 3:  # 如果是RGB图像（CIFAR10）
                img = img.transpose(1, 2, 0)  # 将通道维度从(3, 32, 32)转换为(32, 32, 3)
            else:  # 如果是灰度图像（FashionMNIST）
                img = img.squeeze()
            
            # 确保图像数据在0-1范围内
            img = (img - img.min()) / (img.max() - img.min())
            
            ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
            pred_text = label_to_text_map[preds[i].item()]
            true_text = label_to_text_map[labels[i].item()]
            ax.set_title(f'Pred: {pred_text}\nTrue: {true_text}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()  # 关闭图形以释放内存

    print("可视化结果已保存到 'predictions.png'")

def print_output_distribution(outputs):
    probs = F.softmax(outputs, dim=1)
    avg_probs = probs.mean(dim=0)
    for i, prob in enumerate(avg_probs):
        print(f"Class {i}: {prob.item():.4f}")

def visualize_expert_attention(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        # 获取一个批次的数据
        images, input_ids, attention_mask, _ = next(iter(val_loader))
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # 获取模型输出
        outputs = model(images, input_ids, attention_mask)
        image_expert_outputs = outputs[-2][0]  # 获取图像专家的输出
        
        # 获取注意力图
        attention_map = image_expert_outputs[0].cpu().numpy()  # 取第一个样本
        
        # 计算正确的维度
        img_size = 32  # 原始图像大小
        patch_size = 4  # patch大小
        num_patches = (img_size // patch_size) ** 2  # patch的数量
        
        # 重塑注意力图
        attention_map = attention_map.reshape(int(np.sqrt(num_patches)), int(np.sqrt(num_patches)))
        
        # 可视化
        plt.figure(figsize=(10, 10))
        plt.imshow(attention_map, cmap='viridis')
        plt.colorbar()
        plt.title('Expert Attention Map')
        plt.savefig('expert_attention.png')
        plt.close()
        print("专家注意力图已保存为 'expert_attention.png'")
    
    # 可视化门控网络的分配情况
    plt.figure(figsize=(15, 5))
    
    # 第一层MoE的门控分布
    plt.subplot(1, 2, 1)
    gating_weights = first_gating_output[0].cpu().numpy()
    plt.imshow(gating_weights, aspect='auto', cmap='viridis')
    plt.title('第一层MoE门控分布')
    plt.xlabel('专家')
    plt.ylabel('Token')
    plt.colorbar()
    
    # 第二层MoE的门控分布
    plt.subplot(1, 2, 2)
    gating_weights = second_gating_output[0].cpu().numpy()
    plt.imshow(gating_weights, aspect='auto', cmap='viridis')
    plt.title('第二层MoE门控分布')
    plt.xlabel('专家')
    plt.ylabel('Token')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('gating_distribution.png')
    plt.close()
    
    print("专家注意力可视化结果已保存") 