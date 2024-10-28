import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import label_to_text
import matplotlib
import matplotlib.font_manager as fm

matplotlib.use('Agg') 


def calculate_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def test(model, dataloader, device, label_to_text_map):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            image_feature_vector, text_feature_vector, image_cls, text_cls,  (first_expert_outputs, second_expert_outputs), (first_gating_output, second_gating_output) = model(images, input_ids, attention_mask)
            _, preds = torch.max(image_cls, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    
    print("\n预测结果:")
    print_output_distribution(image_cls)
    for pred, label in zip(all_preds[:10], all_labels[:10]):  # 只打印前10个结果
        pred_text = label_to_text_map[pred]
        true_text = label_to_text_map[label]
        print(f"预测: {pred_text}, 实际: {true_text}")
    
    print(f"\n准确率: {accuracy:.4f}")
    visualize_expert_attention(model, dataloader, device, num_experts=10)
    return accuracy

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
        image_feature_vector, text_feature_vector, image_cls, text_cls,  (first_expert_outputs, second_expert_outputs), (first_gating_output, second_gating_output) = model(images, input_ids, attention_mask)
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



def visualize_expert_attention(model, dataloader, device, num_experts=10):
    model.eval()
    images, input_ids, attention_mask, labels = next(iter(dataloader))
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        image_feature_vector, text_feature_vector, image_cls, text_cls, \
        (first_expert_outputs, second_expert_outputs), (first_gating_output, second_gating_output) = \
            model(images, input_ids, attention_mask)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 可视化图像专家注意力
    batch_size = images.size(0)
    for batch_idx in range(min(3, batch_size)):  # 只显示前3个样本
        fig, axes = plt.subplots(3, num_experts, figsize=(20, 12))
        fig.suptitle(f'样本 {batch_idx + 1} 的专家注意力分布', fontsize=16)
        
        # 显示原图
        img = images[batch_idx].cpu().numpy()
        if img.shape[0] == 3:  # RGB图像
            img = img.transpose(1, 2, 0)
        else:  # 灰度图像
            img = img.squeeze()
        img = (img - img.min()) / (img.max() - img.min())
        
        for expert_idx in range(num_experts):
            axes[0, expert_idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
            axes[0, expert_idx].set_title('原图')
            axes[0, expert_idx].axis('off')
        
        # 第一层MoE的专家注意力
        for expert_idx in range(num_experts):
            expert_output = first_expert_outputs[expert_idx][batch_idx]
            attention_map = expert_output.norm(dim=-1).cpu().numpy()
            attention_map = attention_map.reshape(8, 8)  # 假设patch size为4，32x32图像会得到8x8的注意力图
            
            axes[1, expert_idx].imshow(attention_map, cmap='hot')
            axes[1, expert_idx].set_title(f'专家 {expert_idx}\n第一层')
            axes[1, expert_idx].axis('off')
        
        # 第二层MoE的专家注意力
        for expert_idx in range(num_experts):
            expert_output = second_expert_outputs[expert_idx][batch_idx]
            attention_map = expert_output.norm(dim=-1).cpu().numpy()
            attention_map = attention_map.reshape(8, 8)
            
            axes[2, expert_idx].imshow(attention_map, cmap='hot')
            axes[2, expert_idx].set_title(f'专家 {expert_idx}\n第二层')
            axes[2, expert_idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'expert_attention_sample_{batch_idx}.png')
        plt.close()
    
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
