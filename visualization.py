import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict
import logging
import os
from scipy.ndimage import zoom
from config import VisualizationConfig

# 配置日志
logger = logging.getLogger(__name__)

# 创建可视化目录
os.makedirs(VisualizationConfig.save_dir, exist_ok=True)

def set_chinese_font():
    """设置中文字体"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def visualize_predictions_grid(model, data, labels, device, class_names, num_images=None):
    """可视化预测结果网格"""
    try:
        set_chinese_font()
        if num_images is None:
            num_images = VisualizationConfig.num_images_grid
        model.eval()
        with torch.no_grad():
            # 确保只使用指定数量的图像
            data = data[:num_images]
            labels = labels[:num_images]
            
            # 获取预测结果
            outputs = model(data.to(device))
            predictions = outputs['logits'].argmax(dim=1).cpu()
            
            # 设置图像网格
            n = int(np.ceil(np.sqrt(num_images)))
            fig, axes = plt.subplots(n, n, figsize=(15, 15))
            
            for i, ax in enumerate(axes.flat):
                if i < len(data):
                    # 显示图像
                    img = data[i].cpu()
                    
                    # 反归一化处理
                    if img.shape[0] == 3:  # CIFAR-10
                        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
                        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
                    else:  # FashionMNIST
                        mean = torch.tensor([0.2860]).view(1, 1, 1)
                        std = torch.tensor([0.3530]).view(1, 1, 1)
                    
                    img = img * std + mean
                    img = img.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)  # 确保值在 [0,1] 范围内
                    
                    if img.shape[-1] == 1:
                        img = img.squeeze()
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(img)
                    
                    # 设置标题
                    true_label = class_names[labels[i]]
                    pred_label = class_names[predictions[i]]
                    color = 'green' if labels[i] == predictions[i] else 'red'
                    ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
                
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VisualizationConfig.save_dir, 'predictions_grid.png'), 
                       dpi=VisualizationConfig.dpi)
            plt.close()
            
    except Exception as e:
        logger.error(f"可视化预测结果网格时出错: {str(e)}")

def visualize_expert_regions(model, image, device, class_names):
    """可视化专家处理的图像区域"""
    try:
        set_chinese_font()
        model.eval()
        with torch.no_grad():
            # 使用配置中的图像尺寸
            plt.figure(figsize=VisualizationConfig.expert_regions_fig_size)
            # 确保图像是4D张量 [1, C, H, W]
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(device)
            
            # 获取模型输出，包括注意力权重
            outputs = model(image, return_attention=True)
            attention_weights = outputs.get('attention_weights', [])
            
            if not attention_weights:
                logger.warning("未找到注意力权重")
                return
                
            # 使用最后一层的注意力权重
            attn_weights = attention_weights[-1]  # [B, num_heads, seq_len, seq_len]
            
            # 计算patch大小
            img_size = image.shape[-1]  # 获取输入图像的大小
            if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'patch_size'):
                patch_size = model.patch_embed.patch_size
            else:
                seq_len = attn_weights.shape[-1]
                patch_size = int(img_size / (seq_len ** 0.5))
                logger.info(f"从注意力权重大小推断patch_size为: {patch_size}")
            
            num_patches = (img_size // patch_size) ** 2
            
            # 平均所有头的注意力权重
            attn_map = attn_weights.mean(dim=1)  # [B, seq_len, seq_len]
            attn_map = attn_map[0]  # [seq_len, seq_len]
            
            # 重塑注意力图为方形
            side_length = int(np.sqrt(num_patches))
            attn_map = attn_map.reshape(side_length, side_length).cpu().numpy()
            
            # 使用双线性插值将注意力图调整到原始图像大小
            attn_map = zoom(attn_map, (img_size/side_length, img_size/side_length))
            
            # 显示原始图像
            plt.subplot(131)
            img_np = image[0].cpu().numpy()  # 取第一个样本
            
            # 添加反归一化处理
            if img_np.shape[0] == 3:  # CIFAR-10
                mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
                std = np.array([0.2470, 0.2435, 0.2616]).reshape(3, 1, 1)
                # 反归一化
                img_np = img_np * std + mean
                # 调整维度顺序 [C,H,W] -> [H,W,C]
                img_np = np.transpose(img_np, (1, 2, 0))
                img_np = np.clip(img_np, 0, 1)
                plt.imshow(img_np)
            else:  # FashionMNIST
                mean = np.array([0.2860]).reshape(1, 1, 1)
                std = np.array([0.3530]).reshape(1, 1, 1)
                # 反归一化
                img_np = img_np * std + mean
                img_np = img_np.squeeze()  # 移除通道维度
                img_np = np.clip(img_np, 0, 1)
                plt.imshow(img_np, cmap='gray')
            
            plt.title('原始图像')
            plt.axis('off')
            
            # 显示注意力图
            plt.subplot(132)
            plt.imshow(attn_map, cmap=VisualizationConfig.cmap_attention)
            plt.title('注意力图')
            plt.axis('off')
            
            # 显示叠加后的图像
            plt.subplot(133)
            if img_np.shape[-1] == 3:  # 彩色图像
                plt.imshow(img_np)
            else:  # 灰度图像
                plt.imshow(img_np, cmap='gray')
            plt.imshow(attn_map, cmap=VisualizationConfig.cmap_attention, 
                      alpha=VisualizationConfig.overlay_alpha)
            plt.title('叠加图')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VisualizationConfig.save_dir, 'expert_regions.png'), 
                       bbox_inches='tight', dpi=VisualizationConfig.dpi)
            plt.close()
            
    except Exception as e:
        logger.error(f"可视化专家区域时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_expert_tokens(model, data, labels, device, class_names, num_samples=None):
    """可视化专家处理的token分布"""
    try:
        set_chinese_font()
        if num_samples is None:
            num_samples = VisualizationConfig.num_samples_tokens
        model.eval()
        with torch.no_grad():
            # 准备输入
            if isinstance(data, list):
                data = data[0]
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            data = data[:num_samples].to(device)
            
            # 获取patch嵌入
            x = model.patch_embed(data)
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            x = x + model.pos_embed
            x = model.pos_drop(x)
            
            # 获取每层的路由信息
            router_outputs = []
            for encoder in model.encoders:
                # 获取当前层的路由信息
                shared_router_output = encoder.shared_router(x)
                image_router_output = encoder.image_router(x)
                router_outputs.append({
                    'shared': shared_router_output,
                    'image': image_router_output
                })
                # 通过编码器更新x
                x = encoder(x)['output']
            
            if not router_outputs:
                logger.warning("未找到路由信息")
                return
            
            # 创建图像网格
            num_layers = len(router_outputs)
            fig, axes = plt.subplots(num_layers, 2, figsize=(15, 4*num_layers))
            if num_layers == 1:
                axes = axes.reshape(1, 2)
            
            for i, layer_info in enumerate(router_outputs):
                # 处理共享专家的路由决策
                shared_info = layer_info['shared']
                shared_weights = shared_info.get('weights', shared_info.get('masks', None))
                if shared_weights is not None:
                    if isinstance(shared_weights, torch.Tensor):
                        # 处理不同维度的情况
                        if len(shared_weights.shape) == 4:  # [B, H, N, E]
                            shared_weights = shared_weights.mean(dim=(0, 1))  # 平均batch和head维度
                        elif len(shared_weights.shape) == 3:  # [B, N, E]
                            shared_weights = shared_weights.mean(dim=0)  # 平均batch维度
                        shared_weights = shared_weights.cpu().numpy()
                    
                    # 确保数据是2D的
                    if len(shared_weights.shape) > 2:
                        logger.warning(f"共享专家权重维度过高: {shared_weights.shape}，取平均值")
                        shared_weights = shared_weights.mean(axis=0)
                    
                    sns.heatmap(shared_weights, ax=axes[i, 0], cmap=VisualizationConfig.cmap_heatmap,
                              xticklabels=range(shared_weights.shape[1]),
                              yticklabels=range(shared_weights.shape[0]))
                    axes[i, 0].set_title(f'层 {i+1} 共享专家分配')
                
                # 处理图像专家的路由决策
                image_info = layer_info['image']
                image_weights = image_info.get('weights', image_info.get('masks', None))
                if image_weights is not None:
                    if isinstance(image_weights, torch.Tensor):
                        # 处理不同维度的情况
                        if len(image_weights.shape) == 4:  # [B, H, N, E]
                            image_weights = image_weights.mean(dim=(0, 1))  # 平均batch和head维度
                        elif len(image_weights.shape) == 3:  # [B, N, E]
                            image_weights = image_weights.mean(dim=0)  # 平均batch维度
                        image_weights = image_weights.cpu().numpy()
                    
                    # 确保数据是2D的
                    if len(image_weights.shape) > 2:
                        logger.warning(f"图像专家权重维度过高: {image_weights.shape}，取平均值")
                        image_weights = image_weights.mean(axis=0)
                    
                    sns.heatmap(image_weights, ax=axes[i, 1], cmap=VisualizationConfig.cmap_heatmap,
                              xticklabels=range(image_weights.shape[1]),
                              yticklabels=range(image_weights.shape[0]))
                    axes[i, 1].set_title(f'层 {i+1} 图像专家分配')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VisualizationConfig.save_dir, 'expert_tokens.png'), 
                       bbox_inches='tight', dpi=VisualizationConfig.dpi)
            plt.close()
            
    except Exception as e:
        logger.error(f"可视化专家token时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_router_decisions(model, data, device, num_samples=None):
    """可视化路由决策"""
    try:
        set_chinese_font()
        if num_samples is None:
            num_samples = VisualizationConfig.num_samples_router
        model.eval()
        with torch.no_grad():
            # 准备输入
            if isinstance(data, list):
                data = data[0]
            if len(data.shape) == 3:
                data = data.unsqueeze(0)
            data = data[:num_samples].to(device)
            
            # 获取patch嵌入
            x = model.patch_embed(data)
            x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
            x = x + model.pos_embed
            x = model.pos_drop(x)
            
            # 获取路由信息
            router_outputs = []
            for encoder in model.encoders:
                shared_router_output = encoder.shared_router(x)
                image_router_output = encoder.image_router(x)
                router_outputs.append({
                    'shared': shared_router_output,
                    'image': image_router_output
                })
                x = encoder(x)['output']  # 更新x为编码器的输出
            
            if not router_outputs:
                logger.warning("未找到路由信息")
                return
            
            # 创建图像网格
            num_layers = len(router_outputs)
            fig, axes = plt.subplots(num_layers, 2, figsize=(12, 4*num_layers))
            if num_layers == 1:
                axes = axes.reshape(1, 2)
            
            for i, layer_info in enumerate(router_outputs):
                # 处理共享专家
                shared_info = layer_info['shared']
                shared_weights = shared_info.get('weights', shared_info.get('masks', None))
                if shared_weights is not None:
                    shared_weights = shared_weights.mean(dim=1).cpu().numpy()
                
                # 处理图像专家
                image_info = layer_info['image']
                image_weights = image_info.get('weights', image_info.get('masks', None))
                if image_weights is not None:
                    image_weights = image_weights.mean(dim=1).cpu().numpy()
                
                # 绘制共享专家的路由决策
                if shared_weights is not None:
                    axes[i, 0].imshow(shared_weights, cmap='viridis')
                    axes[i, 0].set_title(f'层 {i+1} 共享专家路由权重')
                    axes[i, 0].axis('off')
                
                # 绘制图像专家的路由决策
                if image_weights is not None:
                    axes[i, 1].imshow(image_weights, cmap='viridis')
                    axes[i, 1].set_title(f'层 {i+1} 图像专家路由权重')
                    axes[i, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(VisualizationConfig.save_dir, 'router_decisions.png'), 
                       bbox_inches='tight', dpi=VisualizationConfig.dpi)
            plt.close()
            
    except Exception as e:
        logger.error(f"可视化路由决策时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_attention(model, data, device, num_samples=None):
    """可视化注意力权重"""
    try:
        set_chinese_font()
        if num_samples is None:
            num_samples = VisualizationConfig.num_samples_attention
        model.eval()
        with torch.no_grad():
            # 准备输入
            if isinstance(data, list):
                data = data[0]
            data = data[:num_samples].to(device)
            
            # 获取模型输出
            outputs = model(data, return_attention=True)
            
            if 'attention_weights' in outputs and outputs['attention_weights']:
                attention_weights = outputs['attention_weights']
                if not isinstance(attention_weights, list):
                    attention_weights = [attention_weights]
                
                if len(attention_weights) > 0:
                    # 创建图像网格
                    num_layers = min(len(attention_weights), 5)  # 最多显示5层
                    if num_samples == 1:
                        fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4))
                        axes = [axes] if num_layers == 1 else axes
                    else:
                        fig, axes = plt.subplots(num_samples, num_layers, figsize=(4*num_layers, 4*num_samples))
                        if num_layers == 1:
                            axes = axes.reshape(-1, 1)
                        elif num_samples == 1:
                            axes = axes.reshape(1, -1)
                    
                    for i in range(min(num_samples, data.shape[0])):
                        for j, attn in enumerate(attention_weights[:num_layers]):
                            if isinstance(attn, torch.Tensor):
                                if len(attn.shape) == 4:  # [B, H, N, N]
                                    attn_map = attn[i].mean(dim=0).cpu().numpy()
                                elif len(attn.shape) == 3:  # [B, N, N]
                                    attn_map = attn[i].cpu().numpy()
                                else:
                                    attn_map = attn.cpu().numpy()
                            else:
                                attn_map = attn
                            
                            if num_samples == 1:
                                ax = axes[j]
                            else:
                                ax = axes[i, j]
                            
                            sns.heatmap(attn_map, ax=ax, cmap='viridis')
                            if i == 0:
                                ax.set_title(f'Layer {j+1}')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(VisualizationConfig.save_dir, 'attention_weights.png'), 
                               dpi=VisualizationConfig.dpi)
                    plt.close()
                else:
                    logger.warning("注意力权重列表为空")
            else:
                logger.warning("未找到注意力权重信息")
                
    except Exception as e:
        logger.error(f"可视化注意力权重时出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def plot_confusion_matrix(predictions, labels, class_names):
    """绘制混淆矩阵"""
    try:
        set_chinese_font()
        fig_size = VisualizationConfig.confusion_matrix_fig_size
        plt.figure(figsize=fig_size)
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        # 绘制混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap=VisualizationConfig.cmap_confusion,
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VisualizationConfig.save_dir, 'confusion_matrix.png'), 
                   dpi=VisualizationConfig.dpi)
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制混淆矩阵时出错: {str(e)}")

def plot_expert_usage(expert_usage):
    """绘制专家使用率分布"""
    try:
        set_chinese_font()
        fig_size = VisualizationConfig.expert_usage_fig_size
        plt.figure(figsize=fig_size)
        plt.bar(range(len(expert_usage)), expert_usage)
        plt.title('Expert Usage Distribution')
        plt.xlabel('Expert ID')
        plt.ylabel('Usage Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VisualizationConfig.save_dir, 'expert_usage.png'), 
                   dpi=VisualizationConfig.dpi)
        plt.close()
        
    except Exception as e:
        logger.error(f"绘制专家使用率分布时出错: {str(e)}")

def visualize_prediction(image, predictions):
    """可视化单张图像的预测结果"""
    try:
        set_chinese_font()
        fig_size = VisualizationConfig.single_prediction_fig_size
        plt.figure(figsize=fig_size)
        
        # 显示图像
        img = image.cpu().permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(img)
        plt.axis('off')
        
        # 添加预测结果文本
        plt.title('Prediction Results', pad=20)
        text = '\n'.join([f"{pred['class']}: {pred['prob']:.2%}" 
                         for pred in predictions[:5]])
        plt.figtext(0.1, -0.1, text, fontsize=12, va='top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VisualizationConfig.save_dir, 'single_prediction.png'), 
                   bbox_inches='tight', dpi=VisualizationConfig.dpi)
        plt.close()
        
    except Exception as e:
        logger.error(f"可视化预测结果时出错: {str(e)}")

def print_prediction_results(predictions):
    """打印预测结果"""
    try:
        print("\n预测结果:")
        for pred in predictions[:5]:
            print(f"{pred['class']}: {pred['prob']:.2%}")
            
    except Exception as e:
        logger.error(f"打印预测结果时出错: {str(e)}") 