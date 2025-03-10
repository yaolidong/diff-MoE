import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional, Union
import argparse
from data_loader import get_dataset_and_loaders
from model import MultiModalMoE
from datasets import get_text_descriptions
from utils import (
    setup_environment, 
    get_model_size, 
    visualize_predictions, 
    plot_confusion_matrix,
    print_model_summary,
    get_device,
    save_confusion_matrix
)
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def validate(model, loader, criterion, device):
    """验证模型
    
    Args:
        model: 要验证的模型
        loader: 数据加载器
        criterion: 损失函数
        device: 运行设备
        
    Returns:
        验证损失和准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                images, input_ids, attention_mask, labels = batch
                images = images.to(device)
                # 将文本输入也传递给模型
                input_ids = input_ids.to(device) if input_ids is not None else None
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
                labels = labels.to(device)
                
                # 将模型设置当前批次的标签以便在forward中使用
                if hasattr(model, 'current_labels'):
                    model.current_labels = labels
                
                # 将图像和文本输入传递给模型
                outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
                
                # 从outputs字典中提取logits
                logits = outputs['logits']
                router_loss = outputs.get('router_loss', 0)
                
                # 使用logits计算损失
                loss = criterion(logits, labels) + router_loss
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # 从outputs字典中提取logits
                logits = outputs['logits']
                router_loss = outputs.get('router_loss', 0)
                
                # 使用logits计算损失
                loss = criterion(logits, labels) + router_loss
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    
    return avg_loss, acc

def test_model(model, test_loader, device, class_names=None, class_descriptions=None):
    """测试模型性能
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 运行设备
        class_names: 类别名称列表（可选）
        class_descriptions: 类别文本描述字典（可选）
        
    Returns:
        包含测试结果的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_router_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                images, input_ids, attention_mask, labels = batch
                images = images.to(device)
                # 将文本输入也传递给模型
                input_ids = input_ids.to(device) if input_ids is not None else None
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
                labels = labels.to(device)
                
                # 将模型设置当前批次的标签以便在forward中使用
                if hasattr(model, 'current_labels'):
                    model.current_labels = labels
                
                # 将图像和文本输入传递给模型
                outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
            
            # 从outputs字典中提取logits
            logits = outputs['logits']
            router_loss = outputs.get('router_loss', 0)
            
            # 计算预测结果
            pred = logits.argmax(dim=1)
            
            # 收集预测和标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 统计正确数
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            # 累计路由损失
            if isinstance(router_loss, torch.Tensor):
                total_router_loss += router_loss.item()
            else:
                total_router_loss += router_loss
    
    # 计算准确率和平均路由损失
    accuracy = 100. * correct / total
    avg_router_loss = total_router_loss / len(test_loader)
    
    # 计算每个类别的准确率
    class_accuracy = {}
    if class_names:
        class_correct = np.zeros(len(class_names))
        class_total = np.zeros(len(class_names))
        
        for pred, label in zip(all_preds, all_labels):
            if label < len(class_names):  # 确保标签在有效范围内
                class_correct[label] += int(pred == label)
                class_total[label] += 1
        
        # 计算每个类别的准确率
        for i, (name, correct, total) in enumerate(zip(class_names, class_correct, class_total)):
            if total > 0:
                class_accuracy[name] = 100. * correct / total
    
    # 计算其他评估指标（如果标签不是多分类，则使用binary或weighted）
    average = 'micro' if len(set(all_labels)) > 2 else 'binary'
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=average, zero_division=0
    )
    
    # 如果提供了保存目录和类名，则保存混淆矩阵
    if class_names:
        cm = confusion_matrix(all_labels, all_preds)
        save_confusion_matrix(cm, class_names, f'visualizations/{test_loader.dataset.__class__.__name__}_confusion_matrix.png')
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'accuracy': accuracy,
        'router_loss': avg_router_loss,
        'class_accuracy': class_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def test_flickr8k(checkpoint_path=None, model=None):
    """测试Flickr8k数据集
    
    Args:
        checkpoint_path: 模型检查点路径，如果为None则创建新模型
        model: 预先创建的模型，如果提供则使用该模型
        
    Returns:
        测试结果字典
    """
    device = setup_environment()
    
    # 加载数据
    train_loader, test_loader, dataset_info = get_dataset_and_loaders('flickr8k', batch_size=128)
    
    # 提取数据集信息
    in_channels = dataset_info['in_channels']
    img_size = dataset_info.get('img_size', 224)
    num_classes = len(dataset_info['class_names'])
    class_names = dataset_info['class_names']
    
    # 使用提供的模型或创建/加载新模型
    if model is not None:
        pass
    else:
        # 创建或加载模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=16,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=384,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=6,
                num_layers=4,
                activation='gelu',
                vocab_size=49408,
                max_text_len=77,
                text_embed_dim=192
            ).to(device)
            
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=16,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=384,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=6,
                num_layers=4,
                activation='gelu',
                vocab_size=49408,
                max_text_len=77,
                text_embed_dim=192
            ).to(device)
    
    # 获取文本描述
    class_descriptions = get_text_descriptions('flickr8k')
    
    # 测试模型
    test_results = test_model(model, test_loader, device, class_names, class_descriptions)
    
    # 打印模型总结
    print_model_summary(model, test_results, class_names)
    
    # 可视化一批数据
    batch_data = next(iter(test_loader))
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        if len(batch_data) == 4:  # 带有文本信息的批次
            test_images, _, _, test_labels = batch_data
        else:  # 普通批次
            test_images, test_labels = batch_data[:2]
            
        visualize_predictions(
            model, 
            test_images[:16], 
            test_labels[:16], 
            device, 
            class_names,
            save_path='visualizations/flickr8k_predictions.png'
        )
        
    return test_results

def test_cifar10(checkpoint_path=None, model=None):
    """测试CIFAR10数据集
    
    Args:
        checkpoint_path: 模型检查点路径，如果为None则创建新模型
        model: 预先创建的模型，如果提供则使用该模型
        
    Returns:
        测试结果字典
    """
    device = setup_environment()
    
    # 加载数据
    train_loader, test_loader, dataset_info = get_dataset_and_loaders('cifar10', batch_size=128)
    
    # 提取数据集信息
    in_channels = dataset_info['in_channels']
    img_size = dataset_info.get('img_size', 32)
    num_classes = len(dataset_info['class_names'])
    class_names = dataset_info['class_names']
    
    # 使用提供的模型或创建/加载新模型
    if model is not None:
        pass
    else:
        # 创建或加载模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=4,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=512,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=8,
                num_layers=6,
                activation='gelu',
                vocab_size=49408,
                max_text_len=32,
                text_embed_dim=128
            ).to(device)
            
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=4,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=512,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=8,
                num_layers=6,
                activation='gelu',
                vocab_size=49408,
                max_text_len=32,
                text_embed_dim=128
            ).to(device)
    
    # 获取文本描述
    class_descriptions = get_text_descriptions('cifar10')
    
    # 测试模型
    test_results = test_model(model, test_loader, device, class_names, class_descriptions)
    
    # 打印模型总结
    print_model_summary(model, test_results, class_names)
    
    # 可视化一批数据
    batch_data = next(iter(test_loader))
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        if len(batch_data) == 4:  # 带有文本信息的批次
            test_images, _, _, test_labels = batch_data
        else:  # 普通批次
            test_images, test_labels = batch_data[:2]
            
        visualize_predictions(
            model, 
            test_images[:16], 
            test_labels[:16], 
            device, 
            class_names,
            save_path='visualizations/cifar10_predictions.png'
        )
        
    return test_results

def test_fashion_mnist(checkpoint_path=None, model=None):
    """测试Fashion-MNIST数据集
    
    Args:
        checkpoint_path: 模型检查点路径，如果为None则创建新模型
        model: 预先创建的模型，如果提供则使用该模型
        
    Returns:
        测试结果字典
    """
    device = setup_environment()
    
    # 加载数据
    train_loader, test_loader, dataset_info = get_dataset_and_loaders('fashion_mnist', batch_size=128)
    
    # 提取数据集信息
    in_channels = dataset_info['in_channels']
    img_size = dataset_info.get('img_size', 28)
    num_classes = len(dataset_info['class_names'])
    class_names = dataset_info['class_names']
    
    # 使用提供的模型或创建/加载新模型
    if model is not None:
        pass
    else:
        # 创建或加载模型
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=4,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=384,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=6,
                num_layers=4,
                activation='gelu',
                vocab_size=49408,
                max_text_len=32,
                text_embed_dim=128
            ).to(device)
            
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=4,
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=384,
                num_shared_experts=4,
                num_modality_specific_experts=2,
                top_k=2,
                dropout=0.1,
                num_heads=6,
                num_layers=4,
                activation='gelu',
                vocab_size=49408,
                max_text_len=32,
                text_embed_dim=128
            ).to(device)
    
    # 获取文本描述
    class_descriptions = get_text_descriptions('fashion_mnist')
    
    # 测试模型
    test_results = test_model(model, test_loader, device, class_names, class_descriptions)
    
    # 打印模型总结
    print_model_summary(model, test_results, class_names)
    
    # 可视化一批数据
    batch_data = next(iter(test_loader))
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        if len(batch_data) == 4:  # 带有文本信息的批次
            test_images, _, _, test_labels = batch_data
        else:  # 普通批次
            test_images, test_labels = batch_data[:2]
            
        visualize_predictions(
            model, 
            test_images[:16], 
            test_labels[:16], 
            device, 
            class_names,
            save_path='visualizations/fashion_mnist_predictions.png'
        )
        
    return test_results

# 在文件中添加一个新的通用test函数
def test(model, test_loader, device, class_names=None):
    """
    通用测试函数，不使用logger记录
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 运行设备
        class_names: 类别名称列表
        
    Returns:
        测试结果字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_router_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 4:
                images, input_ids, attention_mask, labels = batch
                images = images.to(device)
                input_ids = input_ids.to(device) if input_ids is not None else None
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
                labels = labels.to(device)
                
                if hasattr(model, 'current_labels'):
                    model.current_labels = labels
                
                outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                
                if hasattr(model, 'current_labels'):
                    model.current_labels = labels
                
                outputs = model(images)
            
            logits = outputs['logits']
            router_loss = outputs.get('router_loss', 0)
            
            _, predicted = torch.max(logits.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if router_loss:
                total_router_loss += router_loss.item()
    
    # 计算整体准确率
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total
    
    # 计算每个类别的准确率
    class_accuracies = {}
    if class_names:
        for i, name in enumerate(class_names):
            class_indices = (all_labels == i)
            if np.sum(class_indices) > 0:
                class_correct = np.sum((all_preds[class_indices] == i))
                class_accuracy = 100 * class_correct / np.sum(class_indices)
                class_accuracies[name] = class_accuracy
    
    # 计算混淆矩阵(只计算而不可视化)
    num_classes = len(class_names) if class_names else len(np.unique(all_labels))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, label in zip(all_preds, all_labels):
        confusion[label][pred] += 1
    
    # 计算精确度、召回率和F1分数(多分类宏平均)
    precision = 0
    recall = 0
    f1 = 0
    
    for i in range(num_classes):
        true_positive = confusion[i, i]
        false_positive = sum(confusion[:, i]) - true_positive
        false_negative = sum(confusion[i, :]) - true_positive
        
        if true_positive + false_positive > 0:
            precision += true_positive / (true_positive + false_positive)
        
        if true_positive + false_negative > 0:
            recall += true_positive / (true_positive + false_negative)
    
    precision /= num_classes
    recall /= num_classes
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # 返回结果
    results = {
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'predictions': all_preds,
        'labels': all_labels,
        'confusion_matrix': confusion,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'router_loss': total_router_loss / len(test_loader) if total_router_loss else 0
    }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试模型性能')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'fashion_mnist', 'flickr8k'],
                    help='要测试的数据集')
    parser.add_argument('--checkpoint', type=str, default=None,
                    help='模型检查点路径')
    
    args = parser.parse_args()
    
    # 根据选择的数据集执行相应的测试
    if args.dataset == 'cifar10':
        test_cifar10(args.checkpoint)
    elif args.dataset == 'fashion_mnist':
        test_fashion_mnist(args.checkpoint)
    elif args.dataset == 'flickr8k':
        test_flickr8k(args.checkpoint)
    else:
        print(f"不支持的数据集: {args.dataset}") 