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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

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

    if dataset_choice == "1":
        print("您选择了CIFAR10数据集")
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        print("您选择了FashionMNIST数据集")
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

    # 初始化模型
    model = MultiModalMoE(
        img_size=32,
        patch_size=4,
        in_channels=1,
        embed_dim=512,
        num_experts=8,
        top_k=2,
        num_heads=8,
        num_classes=10
    ).to(device)

    if mode_choice == "1":
        print("\n开始训练...")
        # 训练模型
        model = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=1,
            lr=1e-4,
            device=device,
            early_stopping_patience=5
        )
        # 保存模型
        torch.save(model.state_dict(), 'model.pth')
        print("模型已保存至 model.pth")
    else:
        print("\n加载已有模型...")
        try:
            model.load_state_dict(torch.load('model.pth'))
            print("模型加载成功")
        except FileNotFoundError:
            print("未找到已保存的模型文件 model.pth，请先训练模型")
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