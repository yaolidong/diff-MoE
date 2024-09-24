import torch
import torch.nn as nn
import torch.optim as optim
from losses import InfoNCELoss

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self, feature_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        return self.fc(x)

# 生成合成数据
def generate_synthetic_data(batch_size, feature_dim):
    images = torch.randn(batch_size, 128)
    texts = torch.randn(batch_size, 128)
    return images, texts

def main():
    batch_size = 64
    feature_dim = 128
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型和损失函数
    model = SimpleModel(feature_dim).to(device)
    infonce_loss = InfoNCELoss(temperature=0.07).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # 生成合成数据
        images, texts = generate_synthetic_data(batch_size, feature_dim)
        images, texts = images.to(device), texts.to(device)

        optimizer.zero_grad()

        # 前向传播
        image_features = model(images)
        text_features = model(texts)

        # 计算损失
        loss = infonce_loss(image_features, text_features)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    print("训练完成。")

if __name__ == "__main__":
    main()
