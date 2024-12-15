import random
import numpy as np
import torch
import torch.optim as optim
from data_loader import cifar10_label_to_text, get_data_loaders, label_to_text
from model import DualTowerModel
from train import train, EarlyStopping, WarmupLinearScheduler
from test_utils import (
    test,
    visualize_predictions,
    visualize_expert_attention
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 设置随机种子
    set_seed(42)

    # 设置超参数
    epochs = 1
    batch_size = 64
    lr = 0.001
    vocab_size = 30522
    save_path = "best_model.pth"
    warmup_epochs = 5
    patience = 7
    visualize = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = DualTowerModel(vocab_size=vocab_size).to(device)

    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # 加载数据集
    (fashion_train_loader, fashion_test_loader), (cifar_train_loader, cifar_test_loader) = get_data_loaders(batch_size)
    
    # 选择数据集
    print("请选择要使用的数据集：")
    print("1. CIFAR10")
    print("2. FashionMNIST")
    choice = input("请输入选项（1或2）：")
    
    if choice == "1":
        print("您选择了CIFAR10数据集")
        train_loader = cifar_train_loader
        val_loader = cifar_test_loader
        label_to_text_map = cifar10_label_to_text
    elif choice == "2":
        print("您选择了FashionMNIST数据集")
        train_loader = fashion_train_loader
        val_loader = fashion_test_loader
        label_to_text_map = label_to_text
    else:
        print("无效的选择，默认使用FashionMNIST数据集")
        train_loader = fashion_train_loader
        val_loader = fashion_test_loader
        label_to_text_map = label_to_text

    # 训练模型
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        save_path=save_path,
        patience=patience,
        warmup_epochs=warmup_epochs
    )

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    
    # 测试模型
    test(model, val_loader, device, label_to_text_map)

    # 可视化预测结果
    if visualize:
        visualize_predictions(model, val_loader, device, label_to_text_map)
        visualize_expert_attention(model, val_loader, device)

if __name__ == "__main__":
    main() 