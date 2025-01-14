import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MultiModalMoE
from train import train
from test_utils import (
    test_model, print_model_summary, plot_confusion_matrix, 
    plot_expert_usage, visualize_attention,
    predict_single_image, print_prediction_results, visualize_prediction,
    visualize_predictions_grid, visualize_expert_regions, visualize_expert_tokens,
    visualize_router_decisions
)
import os

def main():
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用设备: Apple M1/M2 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用设备: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("使用设备: CPU")

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # 确保model目录存在
    os.makedirs('model', exist_ok=True)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 选择数据集
    print("请选择要使用的数据集：")
    print("1. CIFAR10")
    print("2. FashionMNIST")
    dataset_choice = input("请输入选项（1或2）：")

    # 根据数据集选择设置相应的参数和文件名
    if dataset_choice == "1":
        print("您选择了CIFAR10数据集")
        dataset_name = "cifar10"
        in_channels = 3
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        print("您选择了FashionMNIST数据集")
        dataset_name = "fashion_mnist"
        in_channels = 1
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 选择模式
    print("\n请选择操作模式：")
    print("1. 训练新模型")
    print("2. 加载已有模型")
    mode_choice = input("请输入选项（1或2）：")

    # 构建模型文件路径
    model_path = f'model/model_{dataset_name}.pth'
    best_model_path = f'model/best_model_{dataset_name}.pth'
    checkpoint_path = f'model/checkpoint_{dataset_name}.pth'

    # 初始化模型
    model = MultiModalMoE(
        img_size=32,
        patch_size=4,
        in_channels=in_channels,
        embed_dim=512,
        num_shared_experts=6,
        num_modality_specific_experts=1,
        top_k=2,
        num_heads=8,
        num_layers=6,
        num_classes=10,
        dropout=0.1
    ).to(device)

    if mode_choice == "1":
        print("\n开始训练...")
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
            checkpoint_path=checkpoint_path  # 添加检查点路径
        )
        # 保存最终模型
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存至 {model_path}")
    else:
        print("\n加载已有模型...")
        try:
            # 首先尝试加载最佳模型
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                print(f"已加载最佳模型: {best_model_path}")
            # 如果没有最佳模型，尝试加载普通模型
            elif os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                print(f"已加载模型: {model_path}")
            # 如果都没有，尝试加载检查点
            elif os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"已加载检查点: {checkpoint_path}")
                print(f"检查点轮次: {checkpoint['epoch']}")
            else:
                print(f"未找到任何可用的模型文件，请先训练模型")
                return
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return

    # 选择测试模式
    print("\n请选择测试模式：")
    print("1. 测试整个数据集")
    print("2. 预测单张图像")
    test_mode = input("请输入选项（1或2）：")

    if test_mode == "1":
        print("\n开始测试整个数据集...")
        # 测试模型
        test_results = test_model(model, test_loader, device)
        
        # 打印模型性能总结
        print_model_summary(model, test_results, class_names)
        
        # 绘制混淆矩阵
        plot_confusion_matrix(test_results['predictions'], test_results['labels'], class_names)
        
        # 绘制专家使用率分布
        plot_expert_usage(test_results['expert_usage'][0])
        
        # 获取一批数据用于可视化
        data, labels = next(iter(test_loader))
        
        # 可视化预测结果网格
        print("\n显示预测结果网格...")
        visualize_predictions_grid(model, data, labels, device, class_names)
        
        # 可视化专家处理的图像区域
        print("\n显示专家处理的图像区域...")
        visualize_expert_regions(model, data[0], device, class_names)
        
        # 可视化专家处理的token
        print("\n显示专家处理的token分布...")
        visualize_expert_tokens(model, data, labels, device, class_names)
        
        # 可视化路由决策
        print("\n显示路由决策...")
        visualize_router_decisions(model, data, device)
        
        # 可视化注意力权重
        print("\n显示注意力权重...")
        visualize_attention(model, data, device)
    else:
        print("\n预测单张图像...")
        # 获取一张测试图像
        data, label = next(iter(test_loader))
        image = data[0]  # 取第一张图片
        true_label = class_names[label[0].item()]
        
        # 进行预测
        predictions = predict_single_image(model, image, device, class_names)
        
        # 打印预测结果
        print(f"\n真实类别: {true_label}")
        print_prediction_results(predictions)
        
        # 可视化预测结果
        visualize_prediction(image, predictions)
        
        # 可视化专家处理的图像区域
        print("\n显示专家处理的图像区域...")
        visualize_expert_regions(model, image, device, class_names)
        
        # 可视化路由决策
        print("\n显示路由决策...")
        visualize_router_decisions(model, image.unsqueeze(0), device, num_samples=1)

        print("\n操作完成！")
if __name__ == "__main__":
    main() 