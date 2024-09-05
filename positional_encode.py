import torch
import math

def positional_encoding(batch_size, num_patches, d_model):
    # 创建一个网格，表示每个patch的2D位置
    h = w = int(math.sqrt(num_patches))
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    position = torch.stack((y, x), dim=-1).float()  # shape: [h, w, 2]
    position = position.view(-1, 2)  # shape: [num_patches, 2]

    # 计算除数项
    div_term = torch.exp(torch.arange(0, d_model, 4).float() * -(math.log(10000.0) / d_model))

    # 初始化位置编码
    pe = torch.zeros(num_patches, d_model)

    # 计算正弦和余弦位置编码
    pe[:, 0::4] = torch.sin(position[:, 0].unsqueeze(1) * div_term)
    pe[:, 1::4] = torch.cos(position[:, 0].unsqueeze(1) * div_term)
    pe[:, 2::4] = torch.sin(position[:, 1].unsqueeze(1) * div_term)
    pe[:, 3::4] = torch.cos(position[:, 1].unsqueeze(1) * div_term)

    # 扩展到批次大小
    pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # shape: [batch_size, num_patches, d_model]

    return pe