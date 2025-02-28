import torch
import numpy as np
from typing import Dict, Any, List
import logging
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
    
    # 如果有文本描述，准备文本令牌
    text_tokens = None
    if class_descriptions is not None:
        # 这里简化处理，实际应用中应该使用tokenizer
        # 将文本描述转换为简单的数字序列作为示例
        text_tokens = torch.randint(0, 1000, (len(class_descriptions), 32), device=device)
        logger.info("已准备文本描述令牌")
    
    with torch.no_grad():
        for batch_data in test_loader:
            # 获取数据和标签
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                data, target = batch_data
            else:
                logger.error(f"意外的测试批次数据格式: {type(batch_data)}")
                continue
            
            data, target = data.to(device), target.to(device)
            
            # 前向传播，如果有文本令牌则传入
            if text_tokens is not None:
                outputs = model(data, text_tokens=text_tokens)
            else:
                outputs = model(data)
            
            logits = outputs['logits']
            router_loss = outputs.get('router_loss', 0)
            
            # 计算预测结果
            pred = logits.argmax(dim=1)
            
            # 收集预测和标签
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            # 统计正确数
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 累计路由损失
            if isinstance(router_loss, torch.Tensor):
                total_router_loss += router_loss.item()
            else:
                total_router_loss += router_loss
    
    # 计算准确率和平均路由损失
    accuracy = 100. * correct / total
    avg_router_loss = total_router_loss / len(test_loader)
    
    # 打印测试结果
    logger.info(f"测试结果:")
    logger.info(f"准确率: {accuracy:.2f}%")
    logger.info(f"路由损失: {avg_router_loss:.4f}")
    
    if class_names:
        # 计算每个类别的准确率
        class_correct = np.zeros(len(class_names))
        class_total = np.zeros(len(class_names))
        for pred, label in zip(all_preds, all_labels):
            class_correct[label] += int(pred == label)
            class_total[label] += 1
        
        # 打印每个类别的准确率
        logger.info("\n各类别准确率:")
        for i, (name, correct, total) in enumerate(zip(class_names, class_correct, class_total)):
            accuracy = 100. * correct / total
            logger.info(f"{name}: {accuracy:.2f}%")
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'accuracy': accuracy,
        'router_loss': avg_router_loss
    }

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
        logger.info(f"路由损失: {test_results['router_loss']:.4f}")
        logger.info(f"准确率: {test_results['accuracy']:.2f}%")
        
        # 计算每个类别的准确率
        if len(test_results['predictions']) > 0:
            predictions = test_results['predictions']
            labels = test_results['labels']
            for i, class_name in enumerate(class_names):
                class_mask = (labels == i)
                if class_mask.sum() > 0:
                    class_acc = 100. * (predictions[class_mask] == labels[class_mask]).mean()
                    logger.info(f"{class_name} 准确率: {class_acc:.2f}%")
                    
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