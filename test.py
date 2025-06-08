import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional, Union
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

import torch.nn.functional as F # Added import

def evaluate(model, data_loader, device) -> Dict[str, Any]: # Removed criterion, class_names, class_descriptions
    """评估模型性能 (for KG Alignment Task)
    
    Args:
        model: 要评估的模型
        data_loader: 数据加载器 (expected to yield KGAlignmentDataset batches)
        device: 运行设备
        
    Returns:
        包含评估指标的字典 (loss, hits@1, hits@5, hits@10, mrr)
    """
    model.eval()
        
    total_eval_loss = 0.0
    total_router_loss_eval = 0.0 # Accumulator for router loss during evaluation
    total_alignment_loss_eval = 0.0 # Accumulator for alignment loss during evaluation

    total_hits_at_1 = 0
    total_hits_at_5 = 0
    total_hits_at_10 = 0
    total_mrr = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="评估中"):
            try:
                if not (isinstance(batch, (list, tuple)) and len(batch) == 7):
                    print(f"Warning: Skipping batch of unexpected format in evaluate. Expected 7 items, got {len(batch)}.")
                    continue

                images1, input_ids1, attention_mask1, \
                images2, input_ids2, attention_mask2, _ = batch # true_labels not used for these metrics

                images1 = images1.to(device)
                input_ids1 = input_ids1.to(device) if input_ids1 is not None else None
                attention_mask1 = attention_mask1.to(device) if attention_mask1 is not None else None
                images2 = images2.to(device)
                input_ids2 = input_ids2.to(device) if input_ids2 is not None else None
                attention_mask2 = attention_mask2.to(device) if attention_mask2 is not None else None
                
                outputs = model(images1, input_ids1, attention_mask1,
                                images2, input_ids2, attention_mask2)
                
                embedding1 = outputs['embedding1']
                embedding2 = outputs['embedding2']
                router_loss = outputs.get('router_loss', torch.tensor(0.0, device=device))

                # Alignment Loss Calculation
                embedding1_norm = F.normalize(embedding1, p=2, dim=1)
                embedding2_norm = F.normalize(embedding2, p=2, dim=1)
                
                batch_size = embedding1_norm.size(0)

                margin = 0.2 # Default margin
                if hasattr(data_loader.dataset, 'config') and hasattr(data_loader.dataset.config, 'alignment_margin'):
                    margin = data_loader.dataset.config.alignment_margin

                sim_matrix = torch.matmul(embedding1_norm, embedding2_norm.t())

                batch_alignment_loss = torch.tensor(0.0, device=device)
                for i in range(batch_size):
                    positive_sim = sim_matrix[i, i]
                    
                    negative_sim_e1_vs_e2 = torch.cat((sim_matrix[i, :i], sim_matrix[i, i+1:]))
                    batch_alignment_loss += torch.sum(F.relu(-positive_sim + negative_sim_e1_vs_e2 + margin))
                    
                    negative_sim_e2_vs_e1 = torch.cat((sim_matrix[:i, i], sim_matrix[i+1:, i]))
                    batch_alignment_loss += torch.sum(F.relu(-positive_sim + negative_sim_e2_vs_e1 + margin))
                
                if batch_size > 0:
                    batch_alignment_loss = batch_alignment_loss / (2 * batch_size)
                else:
                    batch_alignment_loss = torch.tensor(0.0, device=device)

                total_alignment_loss_eval += batch_alignment_loss.item()
                total_router_loss_eval += router_loss.item()
                total_eval_loss += (batch_alignment_loss + router_loss).item()

                # Ranking Metrics (Hits@k, MRR) - In-batch
                # For e1_i vs e2_j (ranking e2s for each e1)
                for i in range(batch_size):
                    similarities_for_e1_i = sim_matrix[i, :]
                    sorted_indices = torch.argsort(similarities_for_e1_i, descending=True)

                    rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1

                    if rank <= 1:
                        total_hits_at_1 += 1
                    if rank <= 5:
                        total_hits_at_5 += 1
                    if rank <= 10:
                        total_hits_at_10 += 1
                    total_mrr += 1.0 / rank
                num_samples += batch_size # Each e1 is a sample for ranking e2s

            except Exception as e:
                print(f"处理评估批次时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    avg_eval_loss = total_eval_loss / len(data_loader) if len(data_loader) > 0 else float('inf')
    avg_hits_at_1 = total_hits_at_1 / num_samples if num_samples > 0 else 0.0
    avg_hits_at_5 = total_hits_at_5 / num_samples if num_samples > 0 else 0.0
    avg_hits_at_10 = total_hits_at_10 / num_samples if num_samples > 0 else 0.0
    avg_mrr = total_mrr / num_samples if num_samples > 0 else 0.0
    
    results = {
        'loss': avg_eval_loss,
        'hits@1': avg_hits_at_1,
        'hits@5': avg_hits_at_5,
        'hits@10': avg_hits_at_10,
        'mrr': avg_mrr,
        'router_loss_eval': total_router_loss_eval / len(data_loader) if len(data_loader) > 0 else 0.0,
        'alignment_loss_eval': total_alignment_loss_eval / len(data_loader) if len(data_loader) > 0 else 0.0
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
        # 确定模型参数
        embed_dim = 512 if dataset_type == 'cifar10' else 384
        num_heads = 8 if dataset_type == 'cifar10' else 6
        img_encoder_layers = 6 if dataset_type == 'cifar10' else 4
        text_encoder_layers = 4 if dataset_type == 'cifar10' else 3
        fusion_layers = 3 if dataset_type == 'cifar10' else 2
        max_text_len = 77 if dataset_type == 'flickr8k' else 32
        text_embed_dim = 192 if dataset_type == 'flickr8k' else 128
        
        # 创建模型参数字典
        model_params = {
            'img_size': img_size,
            'patch_size': patch_size,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'embed_dim': embed_dim,
            'num_general_experts': 8,
            'top_k': 2,
            'dropout': 0.1,
            'num_heads': num_heads,
            'img_encoder_layers': img_encoder_layers,
            'text_encoder_layers': text_encoder_layers,
            'fusion_layers': fusion_layers,
            'vocab_size': 50000,
            'max_text_len': max_text_len,
            'text_embed_dim': text_embed_dim,
            'use_checkpoint': False
        }
        
        # 创建模型
        model = MultiModalMoE(**model_params).to(device)
        
        # 如果提供了检查点路径，加载模型权重
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"从 {checkpoint_path} 加载模型权重")
            else:
                model.load_state_dict(checkpoint)
                print(f"从 {checkpoint_path} 加载模型权重")
    
    # 获取文本描述
    class_descriptions = get_text_descriptions(dataset_type)
    
    # 评估模型
    test_results = evaluate(model, test_loader, device, class_names=class_names, class_descriptions=class_descriptions)
    
    # 打印模型总结
    print_model_summary(model, test_results, class_names)
    
    # 可视化预测结果
    batch_data = next(iter(test_loader))
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        if len(batch_data) == 4:
            test_images, input_ids, attention_mask, test_labels = batch_data
        else:
            test_images, test_labels = batch_data[:2]
            
        # 确保可视化目录存在
        os.makedirs('visualizations', exist_ok=True)
            
        visualize_predictions(
            model, 
            test_images[:16], 
            test_labels[:16], 
            device, 
            class_names,
            save_path=f'visualizations/{dataset_type}_predictions.png'
        )
        
        # 尝试可视化专家激活
        try:
            model.eval()
            with torch.no_grad():
                sample_image = test_images[0].unsqueeze(0).to(device)
                
                # 如果有文本输入
                if len(batch_data) == 4:
                    sample_text = input_ids[0].unsqueeze(0).to(device)
                    sample_mask = attention_mask[0].unsqueeze(0).to(device)
                    outputs = model(sample_image, text_tokens=sample_text, attention_mask=sample_mask)
                else:
                    outputs = model(sample_image)
                
                if 'expert_activations' in outputs:
                    expert_act_path = f'visualizations/{dataset_type}_expert_activations.png'
                    visualize_expert_activations(
                        outputs,
                        test_images[0],
                        class_name=class_names[test_labels[0]] if test_labels[0] < len(class_names) else None,
                        save_path=expert_act_path
                    )
                    print(f"专家激活可视化已保存到 {expert_act_path}")
        except Exception as e:
            print(f"可视化专家激活失败: {e}")
    
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