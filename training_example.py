import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import os
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# 导入我们之前创建的模型
from image_classification import create_fashion_mnist_model, create_cifar10_model
from image_captioning import create_flickr8k_model

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================
# Fashion MNIST 训练示例
# ============================
def train_fashion_mnist(epochs=5, batch_size=64, learning_rate=1e-3, weight_decay=1e-4):
    print("=== 训练 Fashion MNIST 模型 ===")
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_dataset = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    model_wrapper = create_fashion_mnist_model()
    model = model_wrapper.model.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练循环
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        router_loss_total = 0.0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            logits = outputs['logits']
            router_loss = outputs['router_loss']
            
            # 计算损失
            classification_loss = criterion(logits, target)
            loss = classification_loss + router_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 优化器步骤
            optimizer.step()
            
            # 累计损失
            train_loss += classification_loss.item()
            router_loss_total += router_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': classification_loss.item(),
                'router_loss': router_loss.item()
            })
        
        # 学习率衰减
        scheduler.step()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        router_loss_total /= len(train_loader)
        
        # 评估模式
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                outputs = model(data)
                logits = outputs['logits']
                
                # 获取预测
                _, predicted = torch.max(logits, 1)
                
                # 统计准确率
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 计算准确率
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Router Loss: {router_loss_total:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "fashion_mnist_best.pth")
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
    
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    return best_accuracy

# ============================
# CIFAR-10 训练示例
# ============================
def train_cifar10(epochs=10, batch_size=32, learning_rate=5e-4, weight_decay=0.05):
    print("=== 训练 CIFAR-10 模型 ===")
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据集
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 添加颜色增强
        transforms.RandomRotation(15),  # 添加随机旋转
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    model_wrapper = create_cifar10_model()
    model = model_wrapper.model.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 使用标签平滑的交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 学习率调度器 - 使用带warmup的余弦退火
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10  # 10%的步数用于warmup
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    best_accuracy = 0.0
    patience = 5  # 早停耐心值
    patience_counter = 0
    
    scaler = torch.cuda.amp.GradScaler()  # 混合精度训练
    
    for epoch in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        router_loss_total = 0.0
        
        # 进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 使用混合精度训练
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = model(data)
                logits = outputs['logits']
                router_loss = outputs['router_loss']
                
                # 计算损失
                classification_loss = criterion(logits, target)
                loss = classification_loss + 0.01 * router_loss  # 降低router_loss的权重
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 累计损失
            train_loss += classification_loss.item()
            router_loss_total += router_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': classification_loss.item(),
                'router_loss': router_loss.item(),
                'lr': scheduler.get_last_lr()[0]
            })
        
        # 计算平均损失
        train_loss /= len(train_loader)
        router_loss_total /= len(train_loader)
        
        # 评估模式
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 前向传播
                outputs = model(data)
                logits = outputs['logits']
                
                # 获取预测
                _, predicted = torch.max(logits, 1)
                
                # 统计准确率
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # 计算准确率
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Router Loss: {router_loss_total:.4f}, Test Accuracy: {accuracy:.2f}%")
        
        # 保存最佳模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "cifar10_best.pth")
            print(f"Saved best model with accuracy: {best_accuracy:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    print(f"最佳测试准确率: {best_accuracy:.2f}%")
    return best_accuracy

# ============================
# Flickr8k 图像摘要生成训练示例
# ============================

# 词汇表类
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, caption_list):
        frequencies = {}
        idx = 4  # 从4开始，因为0-3是特殊标记
        
        for caption in caption_list:
            for word in caption.split():
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        tokenized_text = text.split()
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

# Flickr8k数据集类
class Flickr8kDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        captions_file,
        image_dir,
        transform=None,
        freq_threshold=5,
        max_length=50
    ):
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        
        # 加载标注
        print(f"加载标注文件: {captions_file}")
        self.imgs = []
        self.captions = []
        
        # 读取captions.txt文件 (CSV格式)
        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            
            # 解析每一行
            for row in reader:
                try:
                    if len(row) >= 2:
                        img_name = row[0].strip()
                        caption = row[1].strip()
                        
                        # 构建完整的图像路径
                        img_path = os.path.join(self.image_dir, img_name)
                        
                        if os.path.exists(img_path):
                            self.imgs.append(img_path)
                            self.captions.append(caption.lower())
                        else:
                            print(f"警告: 找不到图像文件: {img_path}")
                except Exception as e:
                    print(f"解析行时出错: {row}, 错误: {str(e)}")
                    continue
        
        print(f"数据集加载完成，共有 {len(self.imgs)} 个图像-文本对")
        if self.imgs:
            print(f"示例图像路径: {self.imgs[0]}")
            print(f"示例标题: {self.captions[0]}")
        
        # 构建词汇表
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)
        print(f"词汇表大小: {len(self.vocab)}")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        caption = self.captions[index]
        
        # 加载图像
        try:
            img = Image.open(img_path).convert("RGB")
            if img.size != (224, 224) and self.transform is None:
                img = img.resize((224, 224))
        except Exception as e:
            print(f"加载图像出错: {img_path}, 错误: {str(e)}")
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            img = self.transform(img)
        
        # 数值化标题
        numericalized_caption = [self.vocab.stoi["<BOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        # 填充或截断到指定长度
        if len(numericalized_caption) < self.max_length:
            numericalized_caption += [self.vocab.stoi["<PAD>"]] * (self.max_length - len(numericalized_caption))
        else:
            numericalized_caption = numericalized_caption[:self.max_length-1] + [self.vocab.stoi["<EOS>"]]
        
        # 创建注意力掩码 (转换为布尔类型)
        attention_mask = torch.tensor([token != self.vocab.stoi["<PAD>"] for token in numericalized_caption], dtype=torch.bool)
        
        return img, torch.tensor(numericalized_caption), attention_mask
        
    def decode_caption(self, caption_ids: np.ndarray) -> str:
        """将数字序列转换回文本
        
        Args:
            caption_ids: 数字序列
            
        Returns:
            解码后的文本
        """
        words = []
        for idx in caption_ids:
            if idx == self.vocab.stoi["<EOS>"]:
                break
            if idx == self.vocab.stoi["<PAD>"] or idx == self.vocab.stoi["<BOS>"]:
                continue
            word = self.vocab.itos.get(idx, "<UNK>")
            words.append(word)
        return " ".join(words)

def train_flickr8k(epochs=5):
    print("开始训练 Flickr8k 图像描述模型...")
    
    # 检查是否有可用的 CUDA 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    captions_file = os.path.join("data", "flickr8k", "captions.txt")
    if not os.path.exists(captions_file):
        print(f"找不到文件: {captions_file}")
        return None
        
    try:
        dataset = Flickr8kDataset(
            captions_file=captions_file,
            image_dir=os.path.join("data", "flickr8k", "images"),
            transform=transform
        )
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return None
        
    # 分割数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器，减小batch_size以提高稳定性
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # 创建模型
    model = create_flickr8k_model()
    model.to(device)
    
    # 优化器和学习率调度器，调整学习率和权重衰减
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,  # 降低学习率
        weight_decay=0.01,  # 增加权重衰减
        betas=(0.9, 0.999),  # 调整beta参数
        eps=1e-8  # 增加数值稳定性
    )
    
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )
    
    # 梯度裁剪值
    max_grad_norm = 0.5  # 降低梯度裁剪阈值
    
    best_val_loss = float('inf')
    best_model = None
    patience = 3  # 早停耐心值
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        valid_batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, captions, attention_mask) in enumerate(progress_bar):
            try:
                # 将数据移动到设备
                images = images.to(device)
                captions = captions.to(device)
                attention_mask = attention_mask.to(device)
                
                # 清零梯度
                optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True以提高性能
                
                # 前向传播
                outputs = model(images, captions, attention_mask)
                
                # 获取损失
                if isinstance(outputs, dict):
                    loss = outputs.get('total_loss', 0)
                    
                    # 检查损失值是否有效
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"警告: 批次 {batch_idx + 1} 出现无效损失值，跳过此批次")
                        continue
                        
                    # 检查损失值是否过大
                    if loss.item() > 100:
                        print(f"警告: 批次 {batch_idx + 1} 损失值过大 ({loss.item():.2f})，跳过此批次")
                        continue
                        
                    # 反向传播
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    # 检查梯度是否有效
                    valid_gradients = True
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"警告: 参数 {name} 的梯度出现无效值")
                                valid_gradients = False
                                break
                    
                    if not valid_gradients:
                        print(f"警告: 批次 {batch_idx + 1} 出现无效梯度，跳过此批次")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # 优化器步骤
                    optimizer.step()
                    
                    # 累计有效损失
                    total_train_loss += loss.item()
                    valid_batch_count += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{(total_train_loss / valid_batch_count):.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
                    
            except Exception as e:
                print(f"处理批次时出错: {str(e)}")
                continue
        
        # 计算平均训练损失
        avg_train_loss = total_train_loss / valid_batch_count if valid_batch_count > 0 else float('inf')
        
        # 验证
        model.eval()
        total_val_loss = 0
        valid_val_batch_count = 0
        
        with torch.no_grad():
            for images, captions, attention_mask in val_loader:
                try:
                    images = images.to(device)
                    captions = captions.to(device)
                    attention_mask = attention_mask.to(device)
                    
                    outputs = model(images, captions, attention_mask)
                    
                    if isinstance(outputs, dict):
                        val_loss = outputs.get('total_loss', 0)
                        
                        if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                            total_val_loss += val_loss.item()
                            valid_val_batch_count += 1
                            
                except Exception as e:
                    print(f"验证时出错: {str(e)}")
                    continue
        
        # 计算平均验证损失
        avg_val_loss = total_val_loss / valid_val_batch_count if valid_val_batch_count > 0 else float('inf')
        
        # 更新学习率
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"平均训练损失: {avg_train_loss:.4f}")
        print(f"平均验证损失: {avg_val_loss:.4f}")
        print(f"学习率: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'best_flickr8k_model.pth')
            print("保存了新的最佳模型")
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= patience:
            print(f"验证损失在 {patience} 轮内没有改善，停止训练")
            break
            
        # 生成一些示例描述
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                try:
                    sample_images, _, _ = next(iter(val_loader))
                    sample_images = sample_images[:3].to(device)
                    generated_output = model.generate(sample_images, max_length=30)
                    
                    if isinstance(generated_output, dict):
                        generated_ids = generated_output.get('generated_ids', None)
                    else:
                        generated_ids = generated_output
                        
                    if generated_ids is not None:
                        for i, ids in enumerate(generated_ids):
                            caption = dataset.decode_caption(ids.cpu().numpy())
                            print(f"示例 {i+1} 生成的描述: {caption}")
                            
                except Exception as e:
                    print(f"生成示例描述时出错: {str(e)}")
    
    return best_val_loss

# ============================
# 主函数
# ============================
if __name__ == "__main__":
    try:
        # 设置随机种子
        set_seed(42)
        
        # 选择要运行的任务
        task = input("选择要运行的任务 (1: Fashion MNIST, 2: CIFAR-10, 3: Flickr8k): ")
        
        if task == "1":
            # 训练Fashion MNIST模型
            train_fashion_mnist(epochs=5)
        elif task == "2":
            # 训练CIFAR-10模型
            train_cifar10(epochs=10)
        elif task == "3":
            # 训练Flickr8k模型
            result = train_flickr8k(epochs=5)
            if result is None:
                print("Flickr8k训练失败，请检查错误信息并修复问题")
        else:
            print("无效选择。请输入1、2或3")
    except Exception as e:
        import traceback
        print(f"程序执行过程中发生错误: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        print("\n请修复上述错误后重试") 