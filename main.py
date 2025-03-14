import os
import datetime  # 添加 datetime 模块导入
import torch
from data_loader import DatasetManager, DatasetType
from model import MultiModalMoE
from train import train
from datasets import (
    CIFAR10_DESCRIPTIONS,
    FASHION_MNIST_DESCRIPTIONS,
    FLICKR8K_DESCRIPTIONS
)
from test import evaluate  # 修改导入语句
from config import CIFAR10Config, FashionMNISTConfig, Flickr8kConfig, GlobalConfig, TrainingConfig, ModelConfig
from utils import (
    setup_environment, plot_training_curves, print_model_summary, 
    set_chinese_font, save_confusion_matrix, visualize_predictions, 
    visualize_expert_regions, visualize_expert_activations,
    visualize_predictions_with_descriptions
)

# 完全禁用PyTorch编译优化
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

def create_model(dataset_info, config: ModelConfig, device):
    """创建模型
    
    Args:
        dataset_info: 数据集信息
        config: 模型配置
        device: 运行设备
        
    Returns:
        创建的模型
    """
    model = MultiModalMoE(
        in_channels=dataset_info['in_channels'],
        img_size=dataset_info['img_size'],
        patch_size=dataset_info.get('patch_size', 4),
        num_classes=dataset_info['num_classes'],
        embed_dim=config.embed_dim,
        num_general_experts=8,  # 设置一般专家数量
        top_k=2,  # 设置top-k值
        dropout=config.dropout,
        num_heads=config.num_heads,
        img_encoder_layers=6,  # 图像MoE编码器层数
        text_encoder_layers=4,  # 文本MoE编码器层数
        fusion_layers=3,  # 融合层数
        device=device,
        vocab_size=config.vocab_size,
        max_text_len=dataset_info.get('max_text_len', 32),
        text_embed_dim=dataset_info.get('text_embed_dim', 128),
        use_checkpoint=config.use_checkpoint
    )
    
    return model.to(device)

def load_checkpoint(model, checkpoint_path, device):
    """加载检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        device: 运行设备
        
    Returns:
        加载了检查点的模型和最佳验证准确率
    """
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return model, 0
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_val_acc = checkpoint.get('best_val_acc', 0)
        print(f"成功加载检查点，最佳验证准确率: {best_val_acc:.2f}%")
        return model, best_val_acc
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}")
        return model, 0

def main():
    """主函数
    """
    # 创建全局配置
    global_config = GlobalConfig()
    # 设置训练配置
    global_config.training = TrainingConfig()
    
    # 创建模型配置
    model_config = ModelConfig()
    
    # 初始化环境
    device = setup_environment(global_config)
    
    # 选择数据集配置 - 默认使用CIFAR10
    dataset_config = CIFAR10Config()
    
    # 创建数据集管理器
    dataset_manager = DatasetManager(
        DatasetType.CIFAR10, 
        dataset_config, 
        batch_size=global_config.training.batch_size
    )
    train_loader, val_loader, test_loader = dataset_manager.get_data_loaders()
    dataset_info = dataset_manager.get_dataset_info()
    
    # 获取文本描述
    class_descriptions = None
    if dataset_config.name == "CIFAR10":
        class_descriptions = CIFAR10_DESCRIPTIONS
    elif dataset_config.name == "FashionMNIST":
        
        class_descriptions = FASHION_MNIST_DESCRIPTIONS
    elif dataset_config.name == "Flickr8k":
        class_descriptions = FLICKR8K_DESCRIPTIONS
    
    # 创建模型
    model = create_model(dataset_info, model_config, device)
    
    # 设置保存路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(global_config.training.checkpoint_dir, f"{dataset_config.name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=global_config.training.learning_rate, 
        weight_decay=global_config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=global_config.training.num_epochs,
        eta_min=global_config.training.learning_rate / 100  # 设置最小学习率，避免降至0
    )
    
    # 训练模型
    model, metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=global_config,
        save_path=save_path,
        class_descriptions=class_descriptions,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # 绘制训练曲线
    plot_path = os.path.join(save_dir, "training_curves.png")
    plot_training_curves(metrics, save_path=plot_path)
    
    # 在测试集上评估
    test_results = evaluate(model, test_loader, device, class_names=dataset_info['class_names'], class_descriptions=class_descriptions)
    print_model_summary(model, test_results, dataset_info['class_names'])
    
    # 设置可视化保存目录
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 设置中文字体
    set_chinese_font()
    
    # 可视化混淆矩阵
    cm_path = os.path.join(vis_dir, "confusion_matrix.png")
    if test_results.get('confusion_matrix') is not None and dataset_info.get('class_names'):
        save_confusion_matrix(
            test_results['confusion_matrix'], 
            dataset_info['class_names'],
            save_path=cm_path
        )
    
    # 获取一批测试数据
    batch_data = next(iter(test_loader))
    if len(batch_data) == 4:  # 多模态数据
        images, input_ids, attention_mask, labels = batch_data
    else:  # 只有图像数据
        images, labels = batch_data
    
    # 可视化预测
    pred_path = os.path.join(vis_dir, "predictions.png")
    visualize_predictions(
        model,
        images[:16],  # 只取前16张图像
        labels[:16],
        device,
        dataset_info['class_names'],
        save_path=pred_path
    )
    
    # 使用中文类别描述可视化预测
    pred_cn_path = os.path.join(vis_dir, "predictions_cn.png")
    visualize_predictions_with_descriptions(
        model,
        images[:16],  # 只取前16张图像
        labels[:16],
        device,
        dataset_info['class_names'],
        class_descriptions=class_descriptions,
        save_path=pred_cn_path
    )
    
    # 可视化专家区域
    try:
        for i in range(min(5, len(images))):  # 可视化前5张图片的专家区域
            expert_path = os.path.join(vis_dir, f"expert_regions_{i}.png")
            visualize_expert_regions(
                model,
                images[i],
                device,
                layer_idx=0,  # 可视化第一层专家
                save_path=expert_path
            )
    except Exception:
        pass
    
    # 运行一次前向传播，尝试获取专家激活
    try:
        model.eval()
        with torch.no_grad():
            test_image = images[0].to(device)
            if len(batch_data) == 4:  # 多模态数据
                outputs = model(
                    test_image.unsqueeze(0), 
                    text_tokens=input_ids[0].unsqueeze(0).to(device),
                    attention_mask=attention_mask[0].unsqueeze(0).to(device)
                )
            else:  # 只有图像数据
                outputs = model(test_image.unsqueeze(0))
                
            # 可视化专家激活
            if 'expert_activations' in outputs:
                expert_act_path = os.path.join(vis_dir, "expert_activations.png")
                visualize_expert_activations(
                    outputs,
                    test_image,
                    class_name=dataset_info['class_names'][labels[0].item()] if labels[0].item() < len(dataset_info['class_names']) else None,
                    save_path=expert_act_path
                )
    except Exception:
        pass
    
    print(f"\n训练和评估完成。结果保存在: {save_dir}")

if __name__ == "__main__":
    main() 
