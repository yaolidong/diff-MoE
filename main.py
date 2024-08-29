import torch
import torch.nn as nn
from model import DualTowerModel
from data_loader import get_data_loaders
from train import train
from test import test, visualize_predictions

def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取数据加载器
    train_dataloader, test_dataloader = get_data_loaders()

    # 初始化模型
    model = DualTowerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

    # 测试模型
    test_accuracy = test(model, test_dataloader, device)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 可视化一些预测结果
    visualize_predictions(model, test_dataloader, device)

if __name__ == "__main__":
    main()
