import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data_loader import label_to_text
import matplotlib
matplotlib.use('Agg') 


def calculate_accuracy(preds, labels):
    return (preds == labels).float().mean().item()

def test(model, dataloader, device):
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
        pred_text = label_to_text[pred]
        true_text = label_to_text[label]
        print(f"预测: {pred_text}, 实际: {true_text}")
    
    print(f"\n准确率: {accuracy:.4f}")
    return accuracy

def visualize_predictions(model, dataloader, device):
    model.eval()
    images, input_ids, attention_mask, labels = next(iter(dataloader))
    images = images.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        image_feature_vector, text_feature_vector, image_cls, text_cls = model(images, input_ids, attention_mask)
        _, preds = torch.max(image_cls, 1)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i].cpu().squeeze()
            if img.dim() == 1:
                img = img.view(28, 28) 
            ax.imshow(img, cmap='gray')
            pred_text = label_to_text[preds[i].item()]
            true_text = label_to_text[labels[i].item()]
            ax.set_title(f'Pred: {pred_text}\nTrue: {true_text}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()  # 关闭图形以释放内存

    print("Visualization results saved to 'predictions.png'")

def print_output_distribution(outputs):
    probs = F.softmax(outputs, dim=1)
    avg_probs = probs.mean(dim=0)
    for i, prob in enumerate(avg_probs):
        print(f"Class {i}: {prob.item():.4f}")

import torch
from model import DualTowerModel
from data_loader import get_data_loaders

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = DualTowerModel(vocab_size=30522).to(device)
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 获取测试数据加载器
    _, test_loader = get_data_loaders(batch_size=64)
    
    # 测试模型性能
    test_accuracy = test(model, test_loader, device)
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device)

