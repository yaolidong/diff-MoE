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
            
            image_feature_vector, text_feature_vector, image_cls, text_cls = model(images, input_ids, attention_mask)
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
        image_feature_vector, text_feature_vector, image_cls, text_cls = model(images, input_ids, attention_mask)
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
