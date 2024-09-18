import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from losses import InfoNCELoss

# 1. 构建 sMoE 模型
class ExpertLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

class sMoEClassifier(nn.Module):
    def __init__(self, num_classes=10, num_experts=4, hidden_dim=128):
        super(sMoEClassifier, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        self.gate = nn.Linear(64 * 7 * 7, num_experts)
        self.experts = nn.ModuleList([ExpertLayer(64 * 7 * 7, hidden_dim) for _ in range(num_experts)])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        features = self.features(x)
        gate_logits = self.gate(features)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        expert_outputs = torch.stack([expert(features) for expert in self.experts])
        combined_output = torch.sum(gate_probs.unsqueeze(-1) * expert_outputs.transpose(0, 1), dim=1)
        
        logits = self.classifier(combined_output)
        return combined_output, logits

# 2. 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sMoEClassifier(num_experts=4).to(device)
infonce_loss = InfoNCELoss(temperature=0.07, device=device)
ce_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            features, logits = model(data)
            
            # 计算 InfoNCE 损失
            infonce = infonce_loss(features,features)
            
            # 计算交叉熵损失
            ce = ce_loss(logits, target)
            
            # 组合损失
            loss = infonce + ce
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    return train_losses

# 4. 评估模型性能
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, logits = model(data)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# 5. 可视化结果
def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

# 运行实验
epochs = 10
train_losses = train(epochs)
accuracy = evaluate()
plot_loss(train_losses)