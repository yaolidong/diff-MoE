import torch
from model import DualTowerModel
from data_loader import get_data_loaders
from train import train
from test import test, visualize_predictions
import platform

def main():
    if platform.system() == 'Darwin':  # macOS
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:  # Windows or other
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    train_dataloader, test_dataloader = get_data_loaders()

    vocab_size = 30522  # 根据实际数据集来设置
    
    model = DualTowerModel(vocab_size=vocab_size, output_dim=1024, n_head=8, num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 将学习率从 1e-3 降低到 1e-4

    num_epochs = 10  # 增加训练轮数以获得更好的结果
    num_epochs = 100  # 增加训练轮数以获得更好的结果
    save_path = "best_model.pth"

    train(model, train_dataloader, optimizer, device, num_epochs, save_path)

    # 修改这里的加载代码
    checkpoint = torch.load(save_path)
    if 'model_version' in checkpoint and checkpoint['model_version'] == '1.1':
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)
    
    model.eval()

    test_accuracy = test(model, test_dataloader, device)
    print(f"测试准确率: {test_accuracy:.4f}")

    visualize_predictions(model, test_dataloader, device)

if __name__ == "__main__":
    main()
