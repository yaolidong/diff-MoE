import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging
from typing import List, Dict, Union, Optional
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, class_names: List[str]) -> None:
    """绘制混淆矩阵
    
    Args:
        predictions: 模型预测结果
        labels: 真实标签
        class_names: 类别名称列表
    """
    try:
        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"绘制混淆矩阵时出错: {str(e)}")

def plot_expert_usage(expert_usage: np.ndarray) -> None:
    """绘制专家使用率分布
    
    Args:
        expert_usage: 专家使用次数数组
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(expert_usage)), expert_usage)
        plt.title('Expert Usage Distribution')
        plt.xlabel('Expert ID')
        plt.ylabel('Usage Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"绘制专家使用率分布时出错: {str(e)}")

def visualize_predictions_grid(
    model: torch.nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 16
) -> None:
    """可视化预测结果网格
    
    Args:
        model: 模型实例
        data: 输入数据批次
        labels: 真实标签
        device: 计算设备
        class_names: 类别名称列表
        num_samples: 要显示的样本数量
    """
    try:
        model.eval()
        with torch.no_grad():
            # 确保数据数量正确
            num_samples = min(num_samples, len(data))
            outputs = model(data[:num_samples].to(device))
            predictions = outputs[0].argmax(dim=1)
            
            # 计算网格大小
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            
            if grid_size == 1:
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            for idx in range(grid_size * grid_size):
                ax = axes[idx // grid_size, idx % grid_size]
                if idx < num_samples:
                    img = data[idx].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                    ax.imshow(img)
                    pred = predictions[idx].item()
                    true = labels[idx].item()
                    color = 'green' if pred == true else 'red'
                    ax.set_title(f'Pred: {class_names[pred]}\nTrue: {class_names[true]}', 
                                color=color, fontsize=8)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化预测结果网格时出错: {str(e)}")

def visualize_expert_regions(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    class_names: List[str]
) -> None:
    """可视化专家处理的图像区域
    
    Args:
        model: 模型实例
        image: 输入图像
        device: 计算设备
        class_names: 类别名称列表
    """
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(image.unsqueeze(0).to(device))
            expert_assignments = outputs[2]
            
            num_experts = len(expert_assignments)
            grid_size = int(np.ceil(np.sqrt(num_experts)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            
            if grid_size == 1:
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            for idx in range(grid_size * grid_size):
                ax = axes[idx // grid_size, idx % grid_size]
                if idx < num_experts:
                    assignment = expert_assignments[idx]
                    assignment_map = assignment.reshape(
                        model.config.image_size // model.config.patch_size,
                        model.config.image_size // model.config.patch_size
                    )
                    im = ax.imshow(assignment_map.cpu(), cmap='hot')
                    ax.set_title(f'Expert {idx}')
                    plt.colorbar(im, ax=ax)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化专家区域时出错: {str(e)}")

def visualize_expert_tokens(
    model: torch.nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    num_samples: int = 4
) -> None:
    """可视化专家处理的token分布
    
    Args:
        model: 模型实例
        data: 输入数据批次
        labels: 真实标签
        device: 计算设备
        class_names: 类别名称列表
        num_samples: 要显示的样本数量
    """
    try:
        model.eval()
        with torch.no_grad():
            num_samples = min(num_samples, len(data))
            outputs = model(data[:num_samples].to(device))
            expert_tokens = outputs[3]
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
            if num_samples == 1:
                axes = np.array([axes])
                
            for idx, (tokens, label) in enumerate(zip(expert_tokens, labels[:num_samples])):
                ax = axes[idx]
                sns.heatmap(tokens.cpu(), ax=ax, cmap='viridis', cbar_kws={'label': 'Token Value'})
                ax.set_title(f'Sample {idx} (Class: {class_names[label]})')
                ax.set_xlabel('Token Position')
                ax.set_ylabel('Expert ID')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化专家token时出错: {str(e)}")

def visualize_router_decisions(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    num_samples: int = 4
) -> None:
    """可视化路由决策
    
    Args:
        model: 模型实例
        data: 输入数据批次
        device: 计算设备
        num_samples: 要显示的样本数量
    """
    try:
        model.eval()
        with torch.no_grad():
            num_samples = min(num_samples, len(data))
            outputs = model(data[:num_samples].to(device))
            router_probs = outputs[4]
            
            fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4*num_samples))
            if num_samples == 1:
                axes = np.array([axes])
                
            for idx, probs in enumerate(router_probs):
                ax = axes[idx]
                sns.heatmap(probs.cpu(), ax=ax, cmap='coolwarm',
                           cbar_kws={'label': 'Routing Probability'})
                ax.set_title(f'Sample {idx} Router Probabilities')
                ax.set_xlabel('Expert ID')
                ax.set_ylabel('Token Position')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化路由决策时出错: {str(e)}")

def visualize_attention(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    num_heads: int = 4
) -> None:
    """可视化注意力权重
    
    Args:
        model: 模型实例
        data: 输入数据
        device: 计算设备
        num_heads: 要显示的注意力头数量
    """
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(data[0:1].to(device))
            attention_weights = outputs[5]
            
            grid_size = int(np.ceil(np.sqrt(num_heads)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
            
            if grid_size == 1:
                axes = np.array([[axes]])
            elif len(axes.shape) == 1:
                axes = axes.reshape(1, -1)
            
            for idx in range(grid_size * grid_size):
                ax = axes[idx // grid_size, idx % grid_size]
                if idx < num_heads:
                    weights = attention_weights[0, idx].cpu()
                    im = sns.heatmap(weights, ax=ax, cmap='viridis',
                                   cbar_kws={'label': 'Attention Weight'})
                    ax.set_title(f'Attention Head {idx}')
                    ax.set_xlabel('Key')
                    ax.set_ylabel('Query')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化注意力权重时出错: {str(e)}")

def predict_single_image(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Union[str, float, np.ndarray]]:
    """预测单张图像
    
    Args:
        model: 模型实例
        image: 输入图像
        device: 计算设备
        class_names: 类别名称列表
        
    Returns:
        包含预测结果的字典
    """
    try:
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            probabilities = F.softmax(output[0], dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            return {
                'class': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }
    except Exception as e:
        logger.error(f"预测单张图像时出错: {str(e)}")
        return None

def print_prediction_results(predictions: Optional[Dict[str, Union[str, float, np.ndarray]]]) -> None:
    """打印预测结果
    
    Args:
        predictions: 预测结果字典
    """
    if predictions is None:
        logger.error("没有有效的预测结果")
        return
        
    try:
        logger.info(f"预测类别: {predictions['class']}")
        logger.info(f"置信度: {predictions['confidence']:.2%}")
        logger.info("\n类别概率分布:")
        for idx, prob in enumerate(predictions['probabilities']):
            logger.info(f"类别 {idx}: {prob:.2%}")
    except Exception as e:
        logger.error(f"打印预测结果时出错: {str(e)}")

def visualize_prediction(
    image: torch.Tensor,
    predictions: Optional[Dict[str, Union[str, float, np.ndarray]]]
) -> None:
    """可视化预测结果
    
    Args:
        image: 输入图像
        predictions: 预测结果字典
    """
    if predictions is None:
        logger.error("没有有效的预测结果")
        return
        
    try:
        plt.figure(figsize=(6, 6))
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.title(f"Predicted: {predictions['class']}\nConfidence: {predictions['confidence']:.2%}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"可视化预测结果时出错: {str(e)}") 