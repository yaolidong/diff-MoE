import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DualTowerModel
from train import train
from test import test
from dataset import CustomDataset  # 假设你有一个自定义的数据集类
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train and test DualTowerModel")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--vocab_size', type=int, default=30522, help='Vocabulary size for text model')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the trained model')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = DualTowerModel(vocab_size=args.vocab_size).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 加载数据集
    train_dataset = CustomDataset(split='train')
    test_dataset = CustomDataset(split='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 训练模型
    train(model, train_loader, optimizer, device, args.epochs, args.save_path)

    # 测试模型
    test(model, test_loader, device)

if __name__ == "__main__":
    main()
