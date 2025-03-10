import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional, Union
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def setup_environment():
    """设置环境
    
    Returns:
        运行设备
    """
    # 设置设备
    device = get_device()
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    return device

def get_device():
    """获取可用的设备"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def get_model_size(model: torch.nn.Module) -> str:
    """获取模型大小
    
    Args:
        model: 模型实例
    
    Returns:
        模型大小的字符串表示
    """
    try:
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_all_mb = (param_size + buffer_size) / 1024**2
        return f"{size_all_mb:.2f} MB"
    except Exception as e:
        return "未知"

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

# 可视化功能
def visualize_predictions(model, images, labels, device, class_names, save_path=None):
    """可视化模型预测结果
    
    Args:
        model: 模型实例
        images: 输入图像张量
        labels: 真实标签张量
        device: 运行设备
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    model.eval()
    
    # 确保图像和标签在正确的设备上
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        
        # 从outputs字典中提取logits
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs
        
        # 计算预测
        _, preds = logits.max(1)
    
    # 转移到CPU进行可视化
    images = images.cpu()
    labels = labels.cpu()
    preds = preds.cpu()
    
    # 创建图像网格
    n_images = min(16, images.shape[0])
    rows = int(np.sqrt(n_images))
    cols = int(np.ceil(n_images / rows))
    
    set_chinese_font()
    plt.figure(figsize=(15, 15))
    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        
        # 反归一化图像
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        color = 'green' if preds[i] == labels[i] else 'red'
        title = f"真: {class_names[labels[i]]}\n预: {class_names[preds[i]]}"
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf()

def plot_confusion_matrix(predictions, labels, class_names, save_path=None):
    """绘制混淆矩阵
    
    Args:
        predictions: 预测结果数组
        labels: 真实标签数组
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    cm = confusion_matrix(labels, predictions)
    
    # 计算每个类别的准确率
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    set_chinese_font()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf()

def save_confusion_matrix(cm, class_names, save_path=None):
    """保存混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        
    Returns:
        图形对象
    """
    # 计算每个类别的准确率
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    set_chinese_font()
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.title('混淆矩阵')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf()

def plot_training_curves(metrics, save_path=None):
    """绘制训练过程的损失和准确率曲线
    
    Args:
        metrics: 包含训练指标的字典
        save_path: 保存路径（可选）
    """
    epochs = list(range(1, len(metrics['train_loss']) + 1))
    
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics['train_loss'], 'b-', label='训练损失')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epochs, metrics['train_acc'], 'b-', label='训练准确率')
    plt.plot(epochs, metrics['val_acc'], 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf()

def print_model_summary(model, test_results, class_names):
    """
    打印模型测试摘要，不使用logger
    
    Args:
        model: 模型
        test_results: 测试结果字典
        class_names: 类别名称列表
    """
    if not test_results:
        print("没有有效的测试结果")
        return
    
    # 计算模型大小
    model_size = get_model_size(model)
    
    print("\n模型性能总结:")
    print(f"准确率: {test_results['accuracy']:.2f}%")
    print(f"路由损失: {test_results['router_loss']:.4f}")
    print(f"精确度: {test_results.get('precision', 0):.4f}")
    print(f"召回率: {test_results.get('recall', 0):.4f}")
    print(f"F1分数: {test_results.get('f1', 0):.4f}")
    print(f"模型大小: {model_size}")
    
    # 打印各类别准确率
    if 'class_accuracies' in test_results and class_names:
        print("\n各类别准确率:")
        for class_name, accuracy in test_results['class_accuracies'].items():
            print(f"{class_name}: {accuracy:.2f}%")

def visualize_expert_regions(model, image, device, layer_idx=0, save_path=None):
    """可视化MoE专家区域
    
    Args:
        model: MoE模型实例
        image: 输入图像张量 [C, H, W]
        device: 运行设备
        layer_idx: 要可视化的层索引
        save_path: 保存路径（可选）
    """
    if len(image.shape) == 3:
        # 添加批次维度
        image = image.unsqueeze(0)
    
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        # 记录专家区域
        expert_activations = []
        
        def hook_fn(module, input, output):
            # 获取router_outputs
            router_outputs = output.get('router_outputs', [])
            if len(router_outputs) > layer_idx:
                expert_activations.append(router_outputs[layer_idx])
        
        # 注册钩子
        handle = model.register_forward_hook(hook_fn)
        
        # 前向传播
        _ = model(image)
        
        # 移除钩子
        handle.remove()
    
    if not expert_activations:
        return None
    
    # 获取专家选择和权重
    expert_activations = expert_activations[0]  # 取出批次中第一个样本的结果
    
    # 绘制专家区域
    set_chinese_font()
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示专家区域
    plt.subplot(1, 2, 2)
    expert_regions = expert_activations['selected_experts'].cpu().numpy()
    expert_weights = expert_activations['combine_weights'].cpu().numpy()
    
    # 重塑为图像形式
    h = w = int(np.sqrt(expert_regions.shape[0]))
    expert_map = np.zeros((h, w))
    
    for i, (experts, weights) in enumerate(zip(expert_regions, expert_weights)):
        row, col = i // w, i % w
        expert_map[row, col] = experts[0]  # 选择第一个激活的专家
    
    # 上采样到与原始图像相同大小
    scale_factor = image.shape[2] / h
    expert_map = zoom(expert_map, scale_factor, order=0)
    
    plt.imshow(expert_map, cmap='viridis')
    plt.colorbar(label='专家索引')
    plt.title("专家分配区域")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf()

def visualize_expert_activations(outputs, image, class_name=None, save_path=None):
    """可视化专家激活区域"""
    # 确保我们有专家信息
    if 'expert_activations' not in outputs or not outputs['expert_activations']:
        return None
        
    # 获取专家激活信息
    expert_activations = outputs['expert_activations']
    
    # 绘制专家激活区域
    set_chinese_font()
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示专家激活区域
    plt.subplot(1, 2, 2)
    expert_map = expert_activations[0].cpu().numpy()
    
    plt.imshow(expert_map, cmap='viridis')
    plt.colorbar(label='专家激活')
    plt.title("专家激活区域")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    
    return plt.gcf() 