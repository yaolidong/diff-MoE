import torch
import numpy as np
from typing import Dict, Any, List
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def test_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """测试模型性能
    
    Args:
        model: 要测试的模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        包含测试结果的字典
    """
    try:
        model.eval()
        all_predictions = []
        all_labels = []
        all_expert_usage = []
        all_router_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                # 将数据移到指定设备
                data, labels = data.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(data)
                logits = outputs[0]
                router_loss = outputs[1]
                expert_assignments = outputs[2]
                
                # 计算损失
                loss = F.cross_entropy(logits, labels) + router_loss
                total_loss += loss.item()
                
                # 获取预测结果
                predictions = logits.argmax(dim=1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_expert_usage.append([assign.sum().cpu().numpy() for assign in expert_assignments])
                all_router_probs.append(outputs[4].cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(test_loader)
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 汇总结果
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'expert_usage': np.array(all_expert_usage),
            'router_probs': np.array(all_router_probs)
        }
        
        return results
    except Exception as e:
        logger.error(f"测试模型时出错: {str(e)}")
        return None

def print_model_summary(model: torch.nn.Module, test_results: Dict[str, Any], class_names: List[str]) -> None:
    """打印模型性能总结
    
    Args:
        model: 模型实例
        test_results: 测试结果字典
        class_names: 类别名称列表
    """
    if test_results is None:
        logger.error("没有有效的测试结果")
        return
        
    try:
        # 打印基本指标
        logger.info("\n模型性能总结:")
        logger.info(f"测试损失: {test_results['loss']:.4f}")
        logger.info(f"准确率: {test_results['accuracy']:.2%}")
        logger.info(f"精确率: {test_results['precision']:.2%}")
        logger.info(f"召回率: {test_results['recall']:.2%}")
        logger.info(f"F1分数: {test_results['f1']:.2%}")
        
        # 打印专家使用情况
        logger.info("\n专家使用情况:")
        expert_usage = test_results['expert_usage'].mean(axis=0)
        for i, usage in enumerate(expert_usage):
            logger.info(f"Expert {i}: {usage:.2f}")
            
        # 打印路由概率统计
        logger.info("\n路由概率统计:")
        router_probs = test_results['router_probs'].mean(axis=(0, 1))
        for i, prob in enumerate(router_probs):
            logger.info(f"Expert {i} 平均路由概率: {prob:.2%}")
            
    except Exception as e:
        logger.error(f"打印模型总结时出错: {str(e)}")

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
        logger.error(f"计算模型大小时出错: {str(e)}")
        return "未知"