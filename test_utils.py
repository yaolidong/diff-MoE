import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import math
import os
from matplotlib.font_manager import FontProperties
import logging
import platform

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_matplotlib_fonts():
    """设置matplotlib的字体，支持中文显示"""
    system = platform.system()
    
    # 根据操作系统选择默认字体
    if system == 'Darwin':  # macOS
        default_fonts = ['PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS']
    elif system == 'Windows':
        default_fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    else:  # Linux或其他
        default_fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
    
    # 添加额外的备选字体
    fallback_fonts = ['DejaVu Sans', 'Liberation Sans']
    all_fonts = default_fonts + fallback_fonts
    
    # 尝试设置字体
    font_found = False
    for font in all_fonts:
        try:
            font_prop = FontProperties(font)
            if font_prop.get_name() != 'DejaVu Sans':  # 验证字体是否真实可用
                plt.rcParams['font.family'] = font
                logger.info(f"成功设置字体: {font}")
                font_found = True
                break
        except Exception as e:
            logger.debug(f"字体 {font} 加载失败: {str(e)}")
            continue
    
    if not font_found:
        logger.warning("未找到合适的中文字体，将使用系统默认字体")
        
    # 设置其他matplotlib参数
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['figure.dpi'] = 300  # 提高默认DPI
    plt.rcParams['savefig.dpi'] = 300  # 提高保存图片的DPI
    plt.rcParams['figure.figsize'] = [10, 6]  # 设置默认图片大小
    plt.rcParams['figure.autolayout'] = True  # 自动调整布局

# 初始化字体设置
setup_matplotlib_fonts()

# 图像处理相关的常量
IMAGE_CMAP = 'gray'  # 灰度图colormap
HEATMAP_CMAP = 'viridis'  # 热力图colormap
FIGURE_DPI = 300  # 图片DPI

def normalize_image(img):
    """
    归一化图像到[0,1]范围
    
    Args:
        img: torch.Tensor或numpy.ndarray类型的图像数据
        
    Returns:
        numpy.ndarray: 归一化后的图像数据
        
    Raises:
        TypeError: 如果输入类型不是torch.Tensor或numpy.ndarray
    """
    if not isinstance(img, (torch.Tensor, np.ndarray)):
        raise TypeError(f"输入图像类型必须是torch.Tensor或numpy.ndarray，而不是{type(img)}")
        
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    
    img_min = img.min()
    img_max = img.max()
    
    if img_max - img_min < 1e-8:
        logger.warning("图像数值范围过小，可能导致归一化结果不准确")
        return np.zeros_like(img)
        
    return (img - img_min) / (img_max - img_min)

def save_figure(name, create_dir=True):
    """
    保存图像到文件
    
    Args:
        name: str, 图像文件名（不包含路径和扩展名）
        create_dir: bool, 是否创建目录，默认为True
        
    Raises:
        IOError: 如果保存失败
    """
    try:
        if create_dir:
            os.makedirs('visualizations', exist_ok=True)
            
        filepath = f'visualizations/visualization_{name}.png'
        plt.savefig(filepath, bbox_inches='tight', dpi=FIGURE_DPI)
        logger.info(f"图像已保存到: {filepath}")
        
    except Exception as e:
        logger.error(f"保存图像失败: {str(e)}")
        raise
    finally:
        plt.close()

def calculate_accuracy(outputs, labels):
    """
    计算分类准确率
    
    Args:
        outputs: dict, 模型输出字典，必须包含'logits'键
        labels: torch.Tensor, 真实标签
        
    Returns:
        float: 分类准确率
        
    Raises:
        KeyError: 如果outputs中没有'logits'键
        ValueError: 如果输入维度不匹配
    """
    if 'logits' not in outputs:
        raise KeyError("模型输出字典中必须包含'logits'键")
        
    logits = outputs['logits']
    if logits.size(0) != labels.size(0):
        raise ValueError(f"logits和labels的批次大小不匹配: {logits.size(0)} vs {labels.size(0)}")
        
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean().item()

def test_model(model, test_loader, device):
    """测试模型性能"""
    model.eval()
    total_loss = 0
    total_router_loss = 0
    total_acc = 0
    all_predictions = []
    all_labels = []
    expert_usage = []
    
    # 每个类别的统计
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            # 计算损失
            loss = F.cross_entropy(outputs['logits'], labels)
            router_loss = outputs['router_loss']
            total_loss += loss.item()
            total_router_loss += router_loss.item()
            
            # 计算准确率
            predictions = torch.argmax(outputs['logits'], dim=1)
            correct = (predictions == labels)
            total_acc += correct.float().mean().item()
            
            # 统计每个类别的准确率
            for label, pred, is_correct in zip(labels, predictions, correct):
                label = label.item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_total[label] += 1
                if is_correct:
                    class_correct[label] += 1
            
            # 收集预测结果
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 收集专家使用情况
            expert_stats = model.get_expert_stats()
            expert_usage.append(expert_stats)
            
    # 计算平均值
    num_batches = len(test_loader)
    avg_loss = total_loss / num_batches
    avg_router_loss = total_router_loss / num_batches
    avg_acc = total_acc / num_batches
    
    # 计算每个类别的准确率
    class_accuracies = {
        label: class_correct[label] / class_total[label]
        for label in class_correct.keys()
    }
    
    return {
        'loss': avg_loss,
        'router_loss': avg_router_loss,
        'accuracy': avg_acc,
        'predictions': all_predictions,
        'labels': all_labels,
        'expert_usage': expert_usage,
        'class_accuracies': class_accuracies,
        'class_total': class_total
    }

def plot_confusion_matrix(predictions, labels, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    save_figure('confusion_matrix')

def plot_expert_usage(expert_usage):
    """绘制专家使用率分布"""
    plt.figure(figsize=(12, 6))
    
    # 图像编码器专家使用率
    plt.subplot(1, 2, 1)
    image_experts = expert_usage['image_encoder']['first_layer']
    plt.bar(range(len(image_experts)), image_experts.cpu().numpy())
    plt.title('图像编码器专家使用率')
    plt.xlabel('专家编号')
    plt.ylabel('使用率')
    
    # 统一编码器专家使用率
    plt.subplot(1, 2, 2)
    unified_experts = expert_usage['cross_modal_layers']['layer_0']
    plt.bar(range(len(unified_experts)), unified_experts.cpu().numpy())
    plt.title('统一编码器专家使用率')
    plt.xlabel('专家编号')
    plt.ylabel('使用率')
    
    plt.tight_layout()
    save_figure('expert_usage')

def visualize_predictions_grid(model, data, labels, device, class_names, num_samples=8):
    """可视化预测结果网格"""
    model.eval()
    with torch.no_grad():
        batch_size = min(num_samples, data.shape[0])
        images = data[:batch_size]
        true_labels = labels[:batch_size]
        
        outputs = model(images.to(device))
        predictions = torch.argmax(outputs['logits'], dim=1)
        
        rows = (batch_size + 3) // 4
        fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 3))
        axes = axes.flatten()
        
        for i in range(batch_size):
            img = images[i].squeeze().cpu()
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            img = normalize_image(img)  # 归一化图像
            axes[i].imshow(img)
            
            pred_label = class_names[predictions[i].item()]
            true_label = class_names[true_labels[i].item()]
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(f'预测: {pred_label}\n真实: {true_label}', color=color)
            axes[i].axis('off')
            
        for i in range(batch_size, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_figure('predictions_grid')

def visualize_expert_regions(model, image, device, class_names):
    """
    可视化每个专家处理的图像区域
    
    Args:
        model: torch.nn.Module, 模型实例
        image: torch.Tensor, 输入图像
        device: torch.device, 计算设备
        class_names: list, 类别名称列表
        
    Raises:
        ValueError: 如果输入参数无效
        RuntimeError: 如果模型处理过程出错
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError("输入图像必须是torch.Tensor类型")
        
    if image.dim() not in [3, 4]:
        raise ValueError(f"输入图像维度必须是3或4，当前是{image.dim()}")
    
    try:
        model.eval()
        with torch.no_grad():
            # 处理输入图像
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            # 获取模型输出
            outputs = model(image)
            expert_activations = outputs.get('expert_outputs', None)
            
            if expert_activations is None:
                logger.warning("专家输出不可用，可能是模型配置问题")
                return
                
            # 获取专家数量和设置图像布局
            num_experts = expert_activations.shape[0]
            num_cols = min(4, num_experts + 1)
            num_rows = (num_experts + num_cols) // num_cols
            
            # 创建图像网格
            fig, axes = plt.subplots(num_rows, num_cols, 
                                   figsize=(num_cols * 4, num_rows * 3),
                                   squeeze=False)
            axes = axes.flatten()
            
            # 显示原始图像
            img = image[0].squeeze().cpu()
            plt.sca(axes[0])
            plt.imshow(img, cmap=IMAGE_CMAP)
            plt.title('原始图像')
            plt.axis('off')
            
            # 计算patch grid的大小
            patch_grid_size = int(math.sqrt(expert_activations.shape[2]))
            
            # 显示每个专家的激活图
            for i in range(num_experts):
                activation = expert_activations[i, 0].cpu()
                activation = activation.reshape(patch_grid_size, patch_grid_size)
                activation = normalize_image(activation)
                
                plt.sca(axes[i+1])
                plt.imshow(activation, cmap=HEATMAP_CMAP)
                plt.title(f'专家 {i} 的处理区域')
                plt.axis('off')
                
            # 隐藏多余的子图
            for i in range(num_experts + 1, len(axes)):
                axes[i].axis('off')
                
            plt.tight_layout()
            save_figure('expert_regions')
            
    except Exception as e:
        logger.error(f"可视化专家区域时发生错误: {str(e)}")
        raise RuntimeError(f"可视化失败: {str(e)}")

def visualize_expert_tokens(model, data, labels, device, class_names):
    """可视化每个专家处理的样本token"""
    model.eval()
    with torch.no_grad():
        batch_size = min(8, data.shape[0])
        data = data[:batch_size].to(device)
        labels = labels[:batch_size]
        
        outputs = model(data)
        gating_output = outputs.get('first_gating_output', None)
        
        if gating_output is None:
            print("门控输出不可用")
            return
            
        # 转换为概率分布
        gating_probs = F.softmax(gating_output, dim=-1) if not torch.is_floating_point(gating_output) else gating_output
        
        # 为每个样本创建一个热力图
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i in range(min(batch_size, 8)):
            sns.heatmap(gating_probs[i].cpu(), 
                       cmap='viridis',
                       xticklabels=[f'专家{j}' for j in range(gating_probs.shape[-1])],
                       yticklabels=[f'Token{j}' for j in range(gating_probs.shape[1])],
                       ax=axes[i])
            axes[i].set_title(f'样本 {i+1} 的专家-Token分配')
            
        plt.tight_layout()
        save_figure('expert_tokens')

def visualize_router_decisions(model, data, device, num_samples=4):
    """可视化路由器选择的专家"""
    model.eval()
    with torch.no_grad():
        batch_size = min(num_samples, data.shape[0])
        images = data[:batch_size].to(device)
        
        outputs = model(images)
        first_router_attn = outputs.get('first_router_attention', None)
        second_router_attn = outputs.get('second_router_attention', None)
        
        if first_router_attn is None or second_router_attn is None:
            print("路由决策信息不可用")
            return
            
        # 创建图像网格
        fig, axes = plt.subplots(batch_size, 2, figsize=(12, 3 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(batch_size):
            # 第一层路由决策
            sns.heatmap(first_router_attn[i].cpu(), 
                       cmap='viridis',
                       xticklabels=[f'专家{k}' for k in range(first_router_attn.shape[-1])],
                       yticklabels=[f'Token{k}' for k in range(first_router_attn.shape[1])],
                       ax=axes[i, 0])
            axes[i, 0].set_title(f'样本 {i+1}, 第一层路由')
            
            # 第二层路由决策
            sns.heatmap(second_router_attn[i].cpu(), 
                       cmap='viridis',
                       xticklabels=[f'专家{k}' for k in range(second_router_attn.shape[-1])],
                       yticklabels=[f'Token{k}' for k in range(second_router_attn.shape[1])],
                       ax=axes[i, 1])
            axes[i, 1].set_title(f'样本 {i+1}, 第二层路由')
                
        plt.tight_layout()
        save_figure('router_decisions')

def visualize_attention(model, data, device):
    """可视化注意力权重"""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        
        # 获取注意力权重
        attn_weights = outputs.get('attention_weights', None)
        if attn_weights is None:
            print("注意力权重不可用")
            return
        
        # 如果是多头注意力取平均
        if len(attn_weights.shape) == 4:  # [batch, heads, seq_len, seq_len]
            attn_weights = attn_weights.mean(dim=1)  # 对所有头取平均
        
        # 取第一个样本的注意力权重
        attn_map = attn_weights[0].cpu()
        
        # 创建图像
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_map, cmap='viridis')
        plt.title('注意力权重可视化')
        plt.xlabel('Query Token')
        plt.ylabel('Key Token')
        
        # 保存图像
        save_figure('attention_weights')
        
        # 打印注意力统计信息
        print(f"\n注意力权重统计:")
        print(f"最小值: {attn_weights.min().item():.4f}")
        print(f"最大值: {attn_weights.max().item():.4f}")
        print(f"平均值: {attn_weights.mean().item():.4f}")
        print(f"标准差: {attn_weights.std().item():.4f}")

def print_model_summary(model, test_results, class_names=None):
    """打印模型性能总结"""
    print("\n模型性能总结:")
    print(f"测试集损失: {test_results['loss']:.4f}")
    print(f"测试集路由损失: {test_results['router_loss']:.4f}")
    print(f"测试集总体准确率: {test_results['accuracy']:.4f}")
    
    # 打印每个类别的准确率
    print("\n各类别准确率:")
    class_accuracies = test_results['class_accuracies']
    class_total = test_results['class_total']
    
    # 按准确率排序
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for label, acc in sorted_classes:
        class_name = class_names[label] if class_names else f"类别 {label}"
        total = class_total[label]
        correct = int(acc * total)
        print(f"{class_name:20s}: {acc:.4f} ({correct}/{total})")
    
    # 打印专家使用情况
    print("\n专家使用情况:")
    expert_usage = test_results['expert_usage'][0]  # 使用第一个batch的数据
    
    print("\n图像编码器专家使用率:")
    image_experts = expert_usage['image_encoder']['first_layer']
    for i, usage in enumerate(image_experts):
        print(f"专家 {i}: {usage:.4f}")
        
    print("\n统一编码器专家使用率:")
    unified_experts = expert_usage['cross_modal_layers']['layer_0']
    for i, usage in enumerate(unified_experts):
        print(f"专家 {i}: {usage:.4f}")
        
    # 打印混淆矩阵
    print("\n混淆矩阵已保存到 visualization_confusion_matrix.png")

def predict_single_image(model, image, device, class_names):
    """对单张图像进行预测"""
    model.eval()
    with torch.no_grad():
        # 确保图像是正确的形状
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加batch维度
        
        # 将图像移到正确的设备
        image = image.to(device)
        
        # 获取预测结果
        outputs = model(image)
        logits = outputs['logits']
        probabilities = F.softmax(logits, dim=1)
        
        # 获取预测类别和概率
        pred_prob, pred_class = torch.max(probabilities, dim=1)
        
        # 获取前3个最可能的类别及其概率
        top3_prob, top3_class = torch.topk(probabilities, 3, dim=1)
        
        # 准备返回结果
        predictions = {
            'predicted_class': class_names[pred_class.item()],
            'confidence': pred_prob.item(),
            'top3_predictions': [
                (class_names[idx.item()], prob.item())
                for idx, prob in zip(top3_class[0], top3_prob[0])
            ]
        }
        
        return predictions

def print_prediction_results(predictions):
    """打印预测结果"""
    print("\n预测结果:")
    print(f"预测类别: {predictions['predicted_class']}")
    print(f"置信度: {predictions['confidence']:.4f}")
    
    print("\n前三可能的类别:")
    for class_name, prob in predictions['top3_predictions']:
        print(f"{class_name}: {prob:.4f}")

def visualize_prediction(image, predictions):
    """可视化预测结果"""
    plt.figure(figsize=(6, 8))
    
    # 显示图像
    plt.subplot(2, 1, 1)
    img = image.squeeze().cpu()
    if img.shape[0] == 3:  # RGB图像
        img = img.permute(1, 2, 0)
    img = normalize_image(img)  # 归一化图像
    plt.imshow(img)
    plt.title(f"预测类别: {predictions['predicted_class']}\n置信度: {predictions['confidence']:.4f}")
    plt.axis('off')
    
    # 显示预测概率条形图
    plt.subplot(2, 1, 2)
    classes, probs = zip(*predictions['top3_predictions'])
    plt.bar(classes, probs)
    plt.title('前三预测概率')
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_figure('single_prediction')