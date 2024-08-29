import torch
from model import DualTowerModel
from data_loader import get_data_loaders
from train import train
from test import test, visualize_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = get_data_loaders()
    model = DualTowerModel(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    save_path = "best_model.pth"

    # 训练模型
    train(model, train_dataloader, optimizer, device, num_epochs, save_path)

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    model.eval()

    # 测试模型
    test_accuracy = test(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # 可视化预测结果
    visualize_predictions(model, test_dataloader, device)

if __name__ == "__main__":
    main()
