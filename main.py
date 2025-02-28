# 在导入所有模块之前设置环境变量和日志级别
import os
import logging
import warnings

# 设置环境变量来控制NumExpr线程数
os.environ["NUMEXPR_MAX_THREADS"] = "16"

# 禁用不必要的日志和警告
logging.getLogger("numexpr").setLevel(logging.WARNING)
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("torchvision").setLevel(logging.WARNING)

# 同时禁用其他可能的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn as nn
from model import MultiModalMoE
from train import train
from test_utils import test_model
from visualization import (
    visualize_predictions_grid, visualize_expert_regions,
    visualize_expert_tokens, visualize_router_decisions,
    visualize_attention, plot_confusion_matrix
)
from config import TrainingConfig, DatasetConfig, ModelConfig
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from data_loader import get_dataset_and_loaders

# 禁用torch.compile错误，允许回退到eager模式
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except ImportError:
    pass

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 提高日志级别到WARNING，减少INFO级别的输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
# 只将主程序的日志级别设为INFO，其他模块保持WARNING级别
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def setup_environment():
    """设置环境"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
        # 性能优化设置
        torch.backends.cudnn.benchmark = TrainingConfig.cudnn_benchmark
        torch.backends.cudnn.deterministic = TrainingConfig.cudnn_deterministic
        
        # 尝试预先分配内存以减少碎片化
        try:
            # 清空缓存并预热
            torch.cuda.empty_cache()
            # 预热GPU - 有助于减少延迟
            dummy = torch.ones(1).cuda()
            del dummy
            logger.info("GPU预热完成")
        except Exception as e:
            logger.warning(f"GPU预热失败: {e}")
    
    return device

def main(dataset_name='cifar10'):
    """主函数"""
    try:
        # 设置环境
        device = setup_environment()
        
        # 获取数据集
        train_loader, test_loader, dataset_info = get_dataset_and_loaders(dataset_name, 
                                                                          TrainingConfig.batch_size)
        
        # 提取数据集信息
        in_channels = dataset_info['in_channels']
        img_size = dataset_info.get('img_size', 32)  # 默认为32
        num_classes = len(dataset_info['class_names'])
        class_names = dataset_info['class_names']
        
        logger.info(f"数据集: {dataset_name}, 图像通道数: {in_channels}, "
                    f"图像大小: {img_size}, 类别数: {num_classes}")
        
        # 创建模型
        try:
            model = MultiModalMoE(
                img_size=img_size,
                patch_size=4,  # 使用固定的patch大小
                in_channels=in_channels,
                num_classes=num_classes,
                embed_dim=256,  # 减小嵌入维度
                num_shared_experts=2,  # 减少共享专家数量
                num_modality_specific_experts=1,  # 减少模态特定专家数量
                top_k=1,  # 减少激活的专家数量
                dropout=0.1,
                num_heads=4,  # 减少注意力头的数量
                num_layers=3,  # 减少层数
                activation='gelu',
                # 添加文本模态相关参数
                vocab_size=1000,  # 词汇表大小
                max_text_len=32,  # 最大文本长度
                text_embed_dim=128  # 文本嵌入维度
            ).to(device)
            logger.info("模型创建成功")
        except Exception as e:
            logger.error(f"模型创建失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
        
        # 完全禁用torch.compile以避免Triton相关错误
        # 根据之前的错误，我们直接跳过编译步骤
        logger.info("跳过torch.compile优化，直接使用未优化模型")
        
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        # 设置损失函数
        criterion = nn.CrossEntropyLoss(label_smoothing=TrainingConfig.label_smoothing)
        
        # 设置优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=TrainingConfig.learning_rate,
            weight_decay=TrainingConfig.weight_decay,
            betas=(0.9, 0.95),  # 调整beta参数，提高动量学习效果
            eps=1e-8  # 增加数值稳定性
        )
        
        # 设置学习率调度器
        total_steps = len(train_loader) * TrainingConfig.num_epochs
        warmup_steps = int(total_steps * TrainingConfig.warmup_ratio)
        
        # 使用OneCycleLR调度器代替余弦调度器，提供更好的学习率调整
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=TrainingConfig.learning_rate,
            total_steps=total_steps,
            pct_start=TrainingConfig.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # 确保checkpoint目录存在
        os.makedirs(TrainingConfig.checkpoint_dir, exist_ok=True)
        
        # 获取GPU显存信息
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.info(f"显存缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        # 训练模型
        try:
            # 获取文本描述
            from label_to_text import get_text_descriptions
            class_descriptions = get_text_descriptions(dataset_name)
            logger.info(f"已加载{len(class_descriptions)}个类别的文本描述")
            
            best_model, train_metrics = train(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                device=device,
                save_path=os.path.join(TrainingConfig.checkpoint_dir, f"{dataset_name}_model.pth"),
                num_epochs=TrainingConfig.num_epochs,
                gradient_clip_val=TrainingConfig.gradient_clip_val,
                early_stopping_patience=TrainingConfig.early_stopping_patience,
                use_amp=False,  # 暂时禁用混合精度训练
                grad_accum_steps=1,  # 减少梯度累积步数
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                class_descriptions=class_descriptions  # 传递文本描述
            )
            
            if best_model is None:
                logger.error("训练失败，无法获取最佳模型")
                return None, None, None
            
            logger.info("训练成功完成")
            
        except Exception as e:
            logger.error(f"训练过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None
        
        # 测试最佳模型
        try:
            test_results = test_model(best_model, test_loader, device, class_names, class_descriptions)
            logger.info("测试成功完成")
        except Exception as e:
            logger.error(f"测试过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            test_results = None
        
        # 提取一些数据用于可视化
        try:
            # 只有在测试成功的情况下才尝试可视化
            if test_results is not None:
                batch_data = next(iter(test_loader))
                if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                    test_images, test_labels = batch_data
                    test_images, test_labels = test_images[:10].to(device), test_labels[:10].to(device)
                    
                    # 执行各种可视化
                    try:
                        visualize_predictions_grid(best_model, test_images, test_labels, device, class_names)
                        logger.info("预测网格可视化完成")
                    except Exception as e:
                        logger.warning(f"预测网格可视化失败: {str(e)}")
                    
                    try:
                        visualize_expert_regions(best_model, test_images[0], device, class_names)
                        logger.info("专家区域可视化完成")
                    except Exception as e:
                        logger.warning(f"专家区域可视化失败: {str(e)}")
                    
                    try:
                        visualize_expert_tokens(best_model, test_images, test_labels, device, class_names)
                        logger.info("专家令牌可视化完成")
                    except Exception as e:
                        logger.warning(f"专家令牌可视化失败: {str(e)}")
                    
                    try:
                        visualize_router_decisions(best_model, test_images, device)
                        logger.info("路由决策可视化完成")
                    except Exception as e:
                        logger.warning(f"路由决策可视化失败: {str(e)}")
                    
                    try:
                        visualize_attention(best_model, test_images, device)
                        logger.info("注意力权重可视化完成")
                    except Exception as e:
                        logger.warning(f"注意力权重可视化失败: {str(e)}")
                    
                    # 生成混淆矩阵
                    try:
                        all_preds = test_results['predictions']
                        all_labels = test_results['labels']
                        plot_confusion_matrix(all_preds, all_labels, class_names)
                        logger.info("混淆矩阵生成完成")
                    except Exception as e:
                        logger.warning(f"混淆矩阵生成失败: {str(e)}")
                else:
                    logger.warning(f"无法执行可视化: 数据格式不正确")
            else:
                logger.warning("由于测试失败，跳过可视化步骤")
        except Exception as e:
            logger.error(f"可视化过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return best_model, train_metrics, test_results
    
    except Exception as e:
        import traceback
        logger.error(f"训练过程中出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

if __name__ == '__main__':
    # 可以选择数据集
    dataset_choices = ['cifar10', 'fashion-mnist']
    print("\n请选择数据集:")
    for i, dataset in enumerate(dataset_choices, 1):
        print(f"{i}. {dataset}")
    
    while True:
        try:
            choice = input("\n请输入数字选择数据集 (1-2): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(dataset_choices):
                dataset_name = dataset_choices[choice_num - 1]
                break
            else:
                print(f"请输入1到{len(dataset_choices)}之间的数字")
        except ValueError:
            print("请输入有效的数字")
    
    print(f"\n已选择数据集: {dataset_name}")
    main(dataset_name) 