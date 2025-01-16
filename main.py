import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import logging
from model import MultiModalMoE
from train import train
from test_utils import *
from data_loader import get_dataset_and_loaders
from visualization import *
import os
from typing import Optional
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置类"""
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    embed_dim: int = 512
    num_shared_experts: int = 4
    num_modality_specific_experts: int = 2
    top_k: int = 2
    num_heads: int = 8
    num_layers: int = 6
    num_classes: int = 10
    dropout: float = 0.1
    activation: str = 'gelu'
    use_bias: bool = True
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

class ModelManager:
    """模型管理类"""
    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device
        
    def create_model(self, in_channels: int) -> MultiModalMoE:
        """创建模型"""
        # 更新配置中的输入通道数
        self.config.in_channels = in_channels
        
        # 创建模型
        model = MultiModalMoE(self.config).to(self.device)
        return model
        
    def load_model(self, model: MultiModalMoE, model_paths: dict) -> Optional[MultiModalMoE]:
        """加载模型"""
        try:
            if os.path.exists(model_paths['best_model']):
                model.load_state_dict(torch.load(model_paths['best_model']))
                logger.info(f"已加载最佳模型: {model_paths['best_model']}")
            elif os.path.exists(model_paths['model']):
                model.load_state_dict(torch.load(model_paths['model']))
                logger.info(f"已加载模型: {model_paths['model']}")
            elif os.path.exists(model_paths['checkpoint']):
                checkpoint = torch.load(model_paths['checkpoint'])
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"已加载检查点: {model_paths['checkpoint']}")
                logger.info(f"检查点轮次: {checkpoint['epoch']}")
            else:
                logger.warning("未找到任何可用的模型文件，请先训练模型")
                return None
            return model
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return None

def setup_device() -> torch.device:
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 设置CUDA设备
        torch.cuda.set_device(0)
        print(f"使用设备: NVIDIA GPU ({torch.cuda.get_device_name(0)})")
        # 设置CUDA随机种子
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: Apple M1/M2 GPU (MPS)")
        # MPS设备不需要特殊设置
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")
    
    # 确保所有操作都是确定性的
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return device

def setup_environment():
    """设置环境"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 确保必要的目录存在
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('bert_cache', exist_ok=True)
    
    # 设置torch的线程数
    torch.set_num_threads(4)  # 避免过度使用CPU线程

def visualize_results(model: MultiModalMoE, test_loader: DataLoader, 
                    device: torch.device, class_names: list):
    """可视化模型结果"""
    # 获取一批数据
    data, labels = next(iter(test_loader))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(test_results['predictions'], test_results['labels'], class_names)
    
    # 绘制专家使用率分布
    plot_expert_usage(test_results['expert_usage'][0])
    
    # 可视化预测结果网格
    logger.info("显示预测结果网格...")
    visualize_predictions_grid(model, data, labels, device, class_names)
    
    # 可视化专家处理的图像区域
    logger.info("显示专家处理的图像区域...")
    visualize_expert_regions(model, data[0], device, class_names)
    
    # 可视化专家处理的token
    logger.info("显示专家处理的token分布...")
    visualize_expert_tokens(model, data, labels, device, class_names)
    
    # 可视化路由决策
    logger.info("显示路由决策...")
    visualize_router_decisions(model, data, device)
    
    # 可视化注意力权重
    logger.info("显示注意力权重...")
    visualize_attention(model, data, device)
    
def visualize_single_image(model: MultiModalMoE, test_loader: DataLoader,
                         device: torch.device, class_names: list):
    """可视化单张图像的预测结果"""
    # 获取一张测试图像
    data, label = next(iter(test_loader))
    image = data[0]  # 取第一张图片
    true_label = class_names[label[0].item()]
    
    # 进行预测
    predictions = predict_single_image(model, image, device, class_names)
    
    # 打印预测结果
    logger.info(f"\n真实类别: {true_label}")
    print_prediction_results(predictions)
    
    # 可视化预测结果
    visualize_prediction(image, predictions)
    
    # 可视化专家处理的图像区域
    logger.info("显示专家处理的图像区域...")
    visualize_expert_regions(model, image, device, class_names)
    
    # 可视化路由决策
    logger.info("显示路由决策...")
    visualize_router_decisions(model, image.unsqueeze(0), device, num_samples=1)
    
    logger.info("操作完成！")

def main():
    try:
        # 设置环境
        setup_environment()
        device = setup_device()
        
        # 选择数据集
        print("请选择要使用的数据集：")
        print("1. CIFAR10")
        print("2. FashionMNIST")
        dataset_choice = input("请输入选项（1或2）：").strip()
        if dataset_choice not in ["1", "2"]:
            logger.error("无效的数据集选择")
            return
        
        dataset_name = 'cifar10' if dataset_choice == "1" else 'fashion_mnist'

        # 获取数据加载器和数据集信息
        try:
            train_loader, test_loader, dataset_info = get_dataset_and_loaders(dataset_name)
        except Exception as e:
            logger.error(f"加载数据集时出错: {str(e)}")
            return
        
        # 创建模型配置
        model_config = ModelConfig(
            in_channels=dataset_info['in_channels'],
            num_classes=len(dataset_info['class_names'])
        )
        
        # 创建模型
        model_manager = ModelManager(model_config, device)
        model = model_manager.create_model(dataset_info['in_channels'])

        # 选择模式
        print("\n请选择操作模式：")
        print("1. 训练新模型")
        print("2. 加载已有模型")
        mode_choice = input("请输入选项（1或2）：").strip()
        if mode_choice not in ["1", "2"]:
            logger.error("无效的操作模式选择")
            return

        if mode_choice == "1":
            logger.info("\n开始训练...")
            try:
                # 训练模型
                model = train(
                    model=model,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    num_epochs=100,
                    lr=1e-4,
                    device=device,
                    early_stopping_patience=10,
                    warmup_epochs=5,
                    weight_decay=0.01,
                    gradient_clip_val=1.0,
                    label_smoothing=0.1,
                    checkpoint_path=dataset_info['model_paths']['checkpoint']
                )
                # 保存最终模型
                torch.save(model.state_dict(), dataset_info['model_paths']['model'])
                logger.info(f"模型已保存至 {dataset_info['model_paths']['model']}")
            except Exception as e:
                logger.error(f"训练模型时出错: {str(e)}")
                return
        else:
            logger.info("\n加载已有模型...")
            model = model_manager.load_model(model, dataset_info['model_paths'])
            if model is None:
                return

        # 选择测试模式
        print("\n请选择测试模式：")
        print("1. 测试整个数据集")
        print("2. 预测单张图像")
        test_mode = input("请输入选项（1或2）：").strip()
        if test_mode not in ["1", "2"]:
            logger.error("无效的测试模式选择")
            return

        if test_mode == "1":
            logger.info("\n开始测试整个数据集...")
            try:
                # 测试模型
                test_results = test_model(model, test_loader, device)
                
                # 打印模型性能总结
                print_model_summary(model, test_results, dataset_info['class_names'])
                
                # 可视化结果
                visualize_results(model, test_loader, device, dataset_info['class_names'])
            except Exception as e:
                logger.error(f"测试模型时出错: {str(e)}")
                return
        else:
            logger.info("\n预测单张图像...")
            try:
                visualize_single_image(model, test_loader, device, dataset_info['class_names'])
            except Exception as e:
                logger.error(f"预测单张图像时出错: {str(e)}")
                return

    except KeyboardInterrupt:
        logger.info("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行时出错: {str(e)}")
    finally:
        # 清理资源
        plt.close('all')  # 关闭所有图形窗口
        if 'model' in locals():
            del model  # 释放模型
        torch.cuda.empty_cache()  # 清理GPU缓存

if __name__ == "__main__":
    main() 