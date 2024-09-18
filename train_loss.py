import torch
import torch.nn as nn
import torch.optim as optim
from losses import CLIPLoss

# 定义简单的模型
class SimpleModel(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        return self.linear(x)

def train():
    feature_dim = 128
    batch_size = 32
    epochs = 50
    loss_fn = CLIPLoss()
    model = SimpleModel(feature_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        # 生成合成数据
        image_input = torch.randn(batch_size, feature_dim)
        text_input = torch.randn(batch_size, feature_dim)
        labels = torch.arange(batch_size)

        # 前向传播
        image_features = model(image_input)
        text_features = model(text_input)

        # 计算损失
        loss = loss_fn(image_features, text_features, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()