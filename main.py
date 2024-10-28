import torch
import torch.optim as optim
import numpy as np
import random
from model import DualTowerModel
from train import train
from test import test, visualize_predictions
from data_loader import get_data_loaders, label_to_text, cifar10_label_to_text

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # 设置随机种子
    set_seed(42)  # 您可以选择任何整数作为种子

    # 设置超参数
    epochs = 10
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
    (fashion_train_loader, fashion_test_loader), (cifar_train_loader, cifar_test_loader) = get_data_loaders(batch_size)
    # 选择数据集
    print("请选择要使用的数据集：")
    print("1. CIFAR10")
    print("2. FashionMNIST")
    choice = input("请输入选项（1或2）：")

    if choice == '1':
        print("您选择了CIFAR10数据集")
        train_loader = cifar_train_loader
        test_loader = cifar_test_loader
        label_to_text_map = cifar10_label_to_text
    elif choice == '2':
        print("您选择了FashionMNIST数据集")
        train_loader = fashion_train_loader
        test_loader = fashion_test_loader
        label_to_text_map = label_to_text
    else:
        print("无效的选择，默认使用FashionMNIST数据集")
        train_loader = fashion_train_loader
        test_loader = fashion_test_loader
        label_to_text_map = label_to_text

    # 训练模型
    train(model, train_loader, optimizer, device, epochs, save_path) 

    # 测试模型
    test(model, test_loader, device, label_to_text_map)

    # 可视化预测结果
    if visualize:
        visualize_predictions(model, test_loader, device, label_to_text_map)

if __name__ == "__main__":
    main()
