import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from losses import CLIPLoss
from train_loss import SimpleModel

def validate_alignment():
    feature_dim = 128
    batch_size = 100
    model = SimpleModel(feature_dim)
    model.eval()

    # 生成合成数据
    image_input = torch.randn(batch_size, feature_dim)
    text_input = torch.randn(batch_size, feature_dim)
    labels = torch.arange(batch_size)

    with torch.no_grad():
        image_features = model(image_input)
        text_features = model(text_input)

    # 降维到2D以便可视化
    features = torch.cat([image_features, text_features], dim=0)
    features_2d = TSNE(n_components=2).fit_transform(features.numpy())
    colors = ['r'] * batch_size + ['b'] * batch_size

    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors)
    plt.title('Feature Alignment Visualization')
    plt.show()

if __name__ == '__main__':
    validate_alignment()