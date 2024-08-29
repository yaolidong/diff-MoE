import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data_loader import label_to_text


def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, texts, labels in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            labels = labels.to(device)
            
            img_first_vec, img_second_vec, img_cls_first, img_cls_second, txt_first_vec, txt_second_vec, txt_cls_first, txt_cls_second = model(images, texts)
            
            similarity = torch.matmul(img_cls_second, txt_cls_second.t())
            _, predicted = similarity.max(1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def visualize_predictions(model, dataloader, device, num_images=5):
    model.eval()
    images, texts, labels = next(iter(dataloader))
    images = images.to(device)
    texts = texts.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        img_embedding, txt_embedding = model(images, labels)
        
        # 计算图像和文本表示之间的相似度
        similarity = torch.matmul(img_embedding, txt_embedding.t())
        
        # 打印相似度矩阵的形状和内容
        print("Similarity matrix shape:", similarity.shape)
        print("Similarity matrix:")
        print(similarity[:num_images, :num_images])
        
        # 获取预测标签
        _, predicted = similarity.max(1)
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i].cpu().view(28, 28), cmap='gray')
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        
        print(f"\nImage {i}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted label: {pred_label}")
        print(f"  True class: {label_to_text[true_label]}")
        print(f"  Predicted class: {label_to_text[pred_label]}")
        
        true_class = label_to_text[true_label]
        pred_class = label_to_text[pred_label]
        
        axes[i].set_title(f"True: {true_class}\nPred: {pred_class}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

    print("\nVisualization saved as 'predictions.png'")
