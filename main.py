import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DualTowerModel
from train import train
from test import test, visualize_predictions
from data_loader import get_data_loaders

def main():
    # 设置超参数
    epochs = 5
    batch_size = 64
    lr = 0.001
    vocab_size = 30522
    save_path = 'model.pth'
    visualize = True  # 是否可视化预测结果

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = DualTowerModel(vocab_size=vocab_size).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 加载数据集
    train_loader, test_loader = get_data_loaders(batch_size)

    # 训练模型
    train(model, train_loader, optimizer, device, epochs, save_path) 

    # 测试模型
    test(model, test_loader, device)

    # 可视化预测结果
    if visualize:
        visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()
