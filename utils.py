import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import random
from typing import Dict, List, Any, Tuple, Optional, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from config import GlobalConfig
import math
import matplotlib.font_manager as fm
from PIL import Image

def setup_environment(config: GlobalConfig):
    """设置环境
    
    Args:
        config: 全局配置对象
        
    Returns:
        运行设备
    """
    # 获取设备
    device = config.device.device
    
    # 设置随机种子
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    return device

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
        return f"计算模型大小时出错: {str(e)}"

def set_chinese_font():
    """设置中文字体"""
    try:
        # 尝试设置微软雅黑字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：未找到中文字体，可能导致中文显示异常")

def print_model_summary(model: nn.Module, results: Dict[str, Any], class_names: Optional[List[str]] = None):
    """打印模型评估结果摘要
    
    Args:
        model: 模型实例
        results: 评估结果字典
        class_names: 类别名称列表
    """
    print("\n" + "="*50)
    print("模型评估结果摘要")
    print("="*50)
    
    # 打印总体指标
    print(f"\n总体准确率: {results['accuracy']:.2f}%")
    print(f"总体损失: {results['loss']:.4f}")
    if 'router_loss' in results:
        print(f"路由损失: {results['router_loss']:.4f}")
    if 'contrastive_loss' in results:
        print(f"对比损失: {results['contrastive_loss']:.4f}")
    
    # 打印每个类别的准确率
    if class_names and 'class_accuracy' in results:
        print("\n各类别准确率:")
        for name, acc in results['class_accuracy'].items():
            print(f"{name}: {acc:.2f}%")
    
    # 打印其他指标
    print(f"\n精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")

def visualize_predictions(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 16
) -> None:
    """可视化模型预测结果
    
    Args:
        model: 模型实例
        images: 输入图像张量 [N, C, H, W]
        labels: 真实标签张量 [N]
        device: 运行设备
        class_names: 类别名称列表
        save_path: 保存路径（可选）
        num_samples: 要显示的样本数量
    """
    model.eval()
    
    # 确保只处理指定数量的样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 获取预测结果
    with torch.no_grad():
        outputs = model(images.to(device))
        logits = outputs['logits']
        predictions = torch.argmax(logits, dim=1)
    
    # 设置中文字体
    set_chinese_font()
    
    # 计算图表网格大小
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # 创建图表
    plt.figure(figsize=(15, 15))
    
    for idx in range(min(num_samples, len(images))):
        plt.subplot(grid_size, grid_size, idx + 1)
        
        # 转换图像格式用于显示
        img = images[idx].cpu()
        if img.shape[0] == 1:  # 灰度图像
            plt.imshow(img.squeeze(), cmap='gray')
        else:  # RGB图像
            img = img.permute(1, 2, 0)  # [C,H,W] -> [H,W,C]
            # 反归一化
            img = img * 0.5 + 0.5  # 假设数据是[-1,1]范围
            img = torch.clamp(img, 0, 1)
            plt.imshow(img)
        
        # 获取预测和真实标签
        pred = predictions[idx].cpu().item()
        true_label = labels[idx].cpu().item()
        
        # 设置标题颜色
        color = 'green' if pred == true_label else 'red'
        
        # 添加标题
        plt.title(f'预测: {class_names[pred]}\n真实: {class_names[true_label]}',
                 color=color, fontsize=8, pad=5)
        
        plt.axis('off')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"预测可视化已保存到: {save_path}")
    
    plt.show()

def save_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
    figsize: tuple = (12, 10),
    normalize: bool = True,
    title: str = "混淆矩阵"
) -> None:
    """保存混淆矩阵可视化
    
    Args:
        confusion_matrix: 混淆矩阵数组
        class_names: 类别名称列表
        save_path: 保存路径
        figsize: 图表大小
        normalize: 是否归一化
        title: 图表标题
    """
    # 设置中文字体
    set_chinese_font()
    
    # 创建图表
    plt.figure(figsize=figsize)
    
    # 计算准确率
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    # 归一化处理
    if normalize:
        row_sums = confusion_matrix.sum(axis=1)
        confusion_matrix_norm = confusion_matrix / row_sums[:, np.newaxis]
        fmt = '.1%'
        data_for_plot = confusion_matrix_norm
    else:
        fmt = 'd'
        data_for_plot = confusion_matrix
    
    # 使用seaborn绘制热力图
    sns.heatmap(data_for_plot, 
                annot=True,
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True)
    
    # 设置标签
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 添加标题和准确率信息
    title_with_acc = f"{title}\n总体准确率: {accuracy:.2%}"
    plt.title(title_with_acc)
    
    # 计算每个类别的准确率
    class_accuracies = np.diag(confusion_matrix) / row_sums
    
    # 在图下方添加每个类别的准确率
    plt.figtext(0.02, 0.02, "各类别准确率:", fontsize=8)
    text_content = ""
    for i, (name, acc) in enumerate(zip(class_names, class_accuracies)):
        text_content += f"{name}: {acc:.1%}  "
        if (i + 1) % 3 == 0:  # 每行显示3个类别
            text_content += "\n"
    plt.figtext(0.02, -0.02, text_content, fontsize=8)
    
    # 调整布局以适应额外的文本
    plt.tight_layout()
    
    # 保存图表
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"混淆矩阵已保存到: {save_path}")
    
    plt.close()

def visualize_expert_activations(
    outputs: Dict[str, Any],
    image: torch.Tensor,
    class_name: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """可视化专家激活情况
    
    Args:
        outputs: 模型输出字典
        image: 输入图像张量
        class_name: 类别名称（可选）
        save_path: 保存路径（可选）
    """
    if 'expert_activations' not in outputs:
        print("警告：未找到专家激活信息")
        return
    
    # 设置中文字体
    set_chinese_font()
    
    # 获取专家激活信息
    activations = outputs['expert_activations']
    
    # 创建图表
    num_encoders = len(activations)
    fig, axes = plt.subplots(num_encoders, 1, figsize=(15, 5*num_encoders))
    if num_encoders == 1:
        axes = [axes]
    
    # 显示原始图像
    img_ax = plt.subplot(num_encoders, 3, 1)
    if image.shape[0] == 1:  # 灰度图像
        img_ax.imshow(image.squeeze().cpu(), cmap='gray')
    else:  # RGB图像
        img = image.permute(1, 2, 0).cpu()  # [C,H,W] -> [H,W,C]
        img = img * 0.5 + 0.5  # 假设数据是[-1,1]范围
        img = torch.clamp(img, 0, 1)
        img_ax.imshow(img)
    img_ax.set_title(f"输入图像 {class_name if class_name else ''}")
    img_ax.axis('off')
    
    # 为每个编码器可视化专家激活
    for i, (encoder_name, encoder_acts) in enumerate(activations.items()):
        if not encoder_acts:  # 跳过空的激活
            continue
            
        ax = axes[i]
        ax.set_title(f"{encoder_name}专家激活")
        
        # 收集所有层的路由概率
        all_probs = []
        for layer_name, layer_outputs in encoder_acts.items():
            if 'router_probs' in layer_outputs:
                probs = layer_outputs['router_probs'].cpu().mean(dim=0)  # 平均池化
                all_probs.append(probs)
        
        if all_probs:
            # 绘制专家使用热力图
            probs_matrix = torch.stack(all_probs)
            sns.heatmap(probs_matrix.numpy(),
                       ax=ax,
                       cmap='YlOrRd',
                       xticklabels=[f'专家{i}' for i in range(probs_matrix.shape[1])],
                       yticklabels=[f'层{i}' for i in range(probs_matrix.shape[0])])
            ax.set_xlabel('专家')
            ax.set_ylabel('层')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"专家激活可视化已保存到: {save_path}")
    
    plt.show()

def visualize_expert_regions(model, image, device, layer_idx=0, save_path=None):
    """可视化模型的专家激活区域
    
    Args:
        model: 模型实例
        image: 输入图像张量 [C, H, W]
        device: 设备
        layer_idx: 要可视化的层索引
        save_path: 保存路径
    """
    model.eval()
    
    # 确保我们有图像
    if image.dim() == 3:
        # 添加批次维度
        image = image.unsqueeze(0)
    
    # 确保图像在正确的设备上
    image = image.to(device)
    
    # 存储专家激活区域
    expert_regions = []
    expert_weights = []
    
    # 定义hook函数
    def hook_fn(module, input, output):
        # 提取一般专家的路由信息
        if isinstance(output, dict) and 'router_probs' in output and 'expert_indices' in output:
            # 获取专家索引和权重
            indices = output['expert_indices']  # [B, S, K]
            weights = output['expert_weights']  # [B, S, K]
            
            # 保存结果
            expert_regions.append(indices.cpu().detach())
            expert_weights.append(weights.cpu().detach())
    
    # 注册hook
    handles = []
    
    # 根据模型架构注册不同的hook
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'layers'):
        # 新架构 - MoE层在image_encoder中
        if layer_idx < len(model.image_encoder.layers):
            layer = model.image_encoder.layers[layer_idx]
            if hasattr(layer, 'router'):
                handle = layer.router.register_forward_hook(hook_fn)
                handles.append(handle)
    elif hasattr(model, 'layers'):
        # 旧架构 - MoE层直接在model中
        if layer_idx < len(model.layers):
            layer = model.layers[layer_idx]
            if hasattr(layer, 'router'):
                handle = layer.router.register_forward_hook(hook_fn)
                handles.append(handle)
    
    # 前向传播
    with torch.no_grad():
        _ = model(image)
    
    # 移除hook
    for handle in handles:
        handle.remove()
    
    # 检查是否有专家激活信息
    if not expert_regions:
        print("没有找到专家激活信息。请确认模型架构和层索引。")
        return None
    
    # 处理专家激活信息 - 只处理第一个样本的结果
    indices = expert_regions[0][0]  # [S, K]
    weights = expert_weights[0][0]  # [S, K]
    
    # 获取原始图像尺寸
    c, h, w = image.shape[1:]
    
    # 获取patch数量
    if hasattr(model, 'patch_embed'):
        patch_size = model.patch_embed.patch_size
        grid_size = model.patch_embed.grid_size
        
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        if isinstance(grid_size, tuple):
            grid_h, grid_w = grid_size
        else:
            # 估计grid大小
            seq_len = indices.shape[0]
            grid_h = grid_w = int(math.sqrt(seq_len))
    else:
        # 估计patch大小和grid大小
        seq_len = indices.shape[0]
        grid_h = grid_w = int(math.sqrt(seq_len))
        patch_size = h // grid_h
    
    # 转换为numpy
    indices_np = indices.numpy()  # [S, K]
    weights_np = weights.numpy()  # [S, K]
    
    # 创建专家分配图
    set_chinese_font()
    plt.figure(figsize=(12, 8))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    plt.title("原始图像")
    plt.axis('off')
    
    # 显示专家分配
    plt.subplot(1, 2, 2)
    
    # 创建专家索引图 - 使用top-1专家
    expert_map = np.zeros((grid_h, grid_w))
    for i in range(seq_len):
        row, col = i // grid_w, i % grid_w
        expert_map[row, col] = indices_np[i, 0]  # 使用top-1专家
    
    # 显示专家分配
    plt.imshow(expert_map, cmap='viridis')
    plt.colorbar(label='专家索引')
    plt.title("专家分配区域")
    plt.axis('off')
    
    # 保存或显示结果
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    
    return expert_map

def visualize_predictions_with_descriptions(model, images, labels, device, class_names, class_descriptions=None, save_path=None):
    """可视化模型预测结果，包含中文类别描述
    
    Args:
        model: 模型实例
        images: 输入图像张量
        labels: 真实标签张量
        device: 运行设备
        class_names: 类别名称列表
        class_descriptions: 类别中文描述字典 {类别索引: 中文描述}
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
    plt.figure(figsize=(16, 16))
    for i in range(n_images):
        plt.subplot(rows, cols, i + 1)
        
        # 反归一化图像
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        plt.imshow(img)
        color = 'green' if preds[i] == labels[i] else 'red'
        
        # 使用中文描述（如果可用）
        true_label = labels[i].item()
        pred_label = preds[i].item()
        
        if class_descriptions and true_label in class_descriptions and pred_label in class_descriptions:
            title = f"真: {class_descriptions[true_label]}\n预: {class_descriptions[pred_label]}"
        else:
            title = f"真: {class_names[true_label]}\n预: {class_names[pred_label]}"
            
        plt.title(title, color=color, fontproperties='SimHei')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return plt.gcf()

def plot_training_curves(metrics, save_path=None):
    """绘制训练曲线
    
    Args:
        metrics: 包含训练指标的字典
        save_path: 保存路径
    """
    if not metrics:
        return
    
    has_router_losses = 'router_z_loss' in metrics and len(metrics['router_z_loss']) > 0
    
    # 设置中文字体
    set_chinese_font()
    
    # 确定子图数量
    n_plots = 3
    if has_router_losses:
        n_plots = 4
    
    # 创建图形
    plt.figure(figsize=(12, 4 * n_plots))
    
    # 绘制训练和验证损失
    plt.subplot(n_plots, 1, 1)
    plt.plot(metrics['train_loss'], label='训练损失')
    plt.plot(metrics['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练和验证准确率
    plt.subplot(n_plots, 1, 2)
    plt.plot(metrics['train_acc'], label='训练准确率')
    plt.plot(metrics['val_acc'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.grid(True)
    
    # 绘制学习率
    plt.subplot(n_plots, 1, 3)
    plt.plot(metrics['lr'])
    plt.title('学习率')
    plt.xlabel('Epoch')
    plt.ylabel('学习率')
    plt.grid(True)
    
    # 如果有路由损失信息，绘制路由损失曲线
    if has_router_losses:
        plt.subplot(n_plots, 1, 4)
        plt.plot(metrics.get('router_z_loss', []), label='Z损失')
        plt.plot(metrics.get('router_balance_loss', []), label='平衡损失')
        plt.plot(metrics.get('cross_modal_loss', []), label='跨模态损失')
        if 'contrastive_loss' in metrics and len(metrics['contrastive_loss']) > 0:
            plt.plot(metrics.get('contrastive_loss', []), label='对比损失')
        plt.title('路由器损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    
    return plt.gcf() 