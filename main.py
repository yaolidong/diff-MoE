import torch
from model import DualTowerModel
from data_loader import get_data_loaders
from train import train
from test import test, visualize_predictions

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = get_data_loaders()

    vocab_size = 30522  # 这里需要根据您的实际数据集来设置
    
    model = DualTowerModel(vocab_size=vocab_size, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    save_path = "best_model.pth"

    train(model, train_dataloader, optimizer, device, num_epochs, save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_accuracy = test(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    visualize_predictions(model, test_dataloader, device)

if __name__ == "__main__":
    main()
