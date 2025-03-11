import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from config import GlobalConfig, CIFAR10Config, FashionMNISTConfig, Flickr8kConfig
from utils import print_model_summary, visualize_predictions, save_confusion_matrix
from data_loader import DatasetManager, DatasetType
from datasets import get_text_descriptions
from model import MultiModalMoE

def setup_environment():
    """设置运行环境"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def evaluate(model, data_loader, device, criterion=None, class_names=None) -> Dict[str, Any]:
    """评估模型性能
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 运行设备
        criterion: 损失函数（可选）
        class_names: 类别名称列表（可选）
        
    Returns:
        包含评估指标的字典
    """
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    all_labels = []
    all_predictions = []
    total_loss = 0
    total_router_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            try:
                # 数据加载和预处理
                if len(batch) == 4:
                    images, input_ids, attention_mask, labels = batch
                    images = images.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    if hasattr(model, 'set_labels'):
                        model.set_labels(labels)
                    
                    outputs = model(images, text_tokens=input_ids, attention_mask=attention_mask)
                else:
                    images, labels = batch
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    if hasattr(model, 'set_labels'):
                        model.set_labels(labels)
                        
                    outputs = model(images)
                
                # 计算损失和预测
                logits = outputs['logits']
                router_loss = outputs.get('router_loss', 0)
                loss = criterion(logits, labels) + router_loss
                
                _, predicted = torch.max(logits.data, 1)
                
                # 收集结果
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item()
                if isinstance(router_loss, torch.Tensor):
                    total_router_loss += router_loss.item()
                else:
                    total_router_loss += router_loss
                valid_batches += 1
                
            except Exception as e:
                logging.error(f"处理批次时出错: {str(e)}")
                continue
    
    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算整体准确率
    correct = (all_predictions == all_labels).sum()
    total = len(all_labels)
    accuracy = 100 * correct / total
    
    # 计算每个类别的准确率
    class_accuracies = {}
    if class_names:
        for i, name in enumerate(class_names):
            class_indices = (all_labels == i)
            if np.sum(class_indices) > 0:
                class_correct = np.sum((all_predictions[class_indices] == i))
                class_accuracy = 100 * class_correct / np.sum(class_indices)
                class_accuracies[name] = class_accuracy
    
    # 计算其他指标
    average = 'micro' if len(set(all_labels)) > 2 else 'binary'
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=average, zero_division=0
    )
    
    # 计算混淆矩阵
    if class_names:
        cm = confusion_matrix(all_labels, all_predictions)
    else:
        cm = None
    
    # 返回结果
    results = {
        'accuracy': accuracy,
        'loss': total_loss / valid_batches if valid_batches > 0 else float('inf'),
        'router_loss': total_router_loss / valid_batches if valid_batches > 0 else 0,
        'class_accuracy': class_accuracies,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'confusion_matrix': cm
    }
    
    return results

def test_dataset(dataset_type: str, checkpoint_path: str = None, model = None):
    """测试指定数据集
    
    Args:
        dataset_type: 数据集类型 ('cifar10', 'fashion_mnist', 'flickr8k')
        checkpoint_path: 模型检查点路径（可选）
        model: 预先创建的模型（可选）
        
    Returns:
        测试结果字典
    """
    device = setup_environment()
    
    # 选择数据集配置
    if dataset_type == 'cifar10':
        config = CIFAR10Config()
        dataset_enum = DatasetType.CIFAR10
    elif dataset_type == 'fashion_mnist':
        config = FashionMNISTConfig()
        dataset_enum = DatasetType.FASHION_MNIST
    elif dataset_type == 'flickr8k':
        config = Flickr8kConfig()
        dataset_enum = DatasetType.FLICKR8K
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")
    
    # 创建数据集管理器
    dataset_manager = DatasetManager(dataset_enum, config, batch_size=128)
    _, _, test_loader = dataset_manager.get_data_loaders()
    dataset_info = dataset_manager.get_dataset_info()
    
    # 获取数据集信息
    in_channels = dataset_info['in_channels']
    img_size = dataset_info['img_size']
    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    patch_size = dataset_info.get('patch_size', 4)
    
    # 使用提供的模型或创建新模型
    if model is None:
        # 创建模型配置
        model_params = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'embed_dim': 512 if dataset_type == 'cifar10' else 384,
            'num_shared_experts': 4,
            'num_modality_specific_experts': 2,
            'top_k': 2,
            'dropout': 0.1,
            'num_heads': 8 if dataset_type == 'cifar10' else 6,
            'num_layers': 6 if dataset_type == 'cifar10' else 4,
            'activation': 'gelu',
            'vocab_size': 49408,
            'max_text_len': 77 if dataset_type == 'flickr8k' else 32,
            'text_embed_dim': 192 if dataset_type == 'flickr8k' else 128
        }
        
        # 创建模型
        model = MultiModalMoE(**model_params).to(device)
        
        # 如果提供了检查点路径，加载模型权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 获取文本描述
    class_descriptions = get_text_descriptions(dataset_type)
    
    # 评估模型
    test_results = evaluate(model, test_loader, device, class_names=class_names)
    
    # 打印模型总结
    print_model_summary(model, test_results, class_names)
    
    # 可视化预测结果
    batch_data = next(iter(test_loader))
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        if len(batch_data) == 4:
            test_images, _, _, test_labels = batch_data
        else:
            test_images, test_labels = batch_data[:2]
            
        visualize_predictions(
            model, 
            test_images[:16], 
            test_labels[:16], 
            device, 
            class_names,
            save_path=f'visualizations/{dataset_type}_predictions.png'
        )
    
    return test_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='测试模型性能')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['cifar10', 'fashion_mnist', 'flickr8k'],
                       help='要测试的数据集')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')
    
    args = parser.parse_args()
    test_dataset(args.dataset, args.checkpoint) 