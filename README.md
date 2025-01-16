# Diff-MoE: 多模态混合专家模型

这是一个基于PyTorch实现的多模态混合专家模型，支持图像分类任务。

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- CUDA (可选，用于GPU加速)

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/diff-MoE.git
cd diff-MoE
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行训练：
```bash
python main.py
```

2. 选择数据集：
- 输入1: CIFAR10
- 输入2: Fashion-MNIST

3. 选择操作模式：
- 输入1: 训练新模型
- 输入2: 加载已有模型

4. 选择测试模式：
- 输入1: 测试整个数据集
- 输入2: 预测单张图像

## 特性

- 支持多种数据集（CIFAR10, Fashion-MNIST）
- 混合专家路由机制
- 支持CPU、NVIDIA GPU和Apple Silicon GPU
- 自动模型检查点保存
- 可视化预测结果和专家分配

## 目录结构

```
diff-MoE/
├── model/              # 保存模型文件
├── data/               # 数据集目录
├── bert_cache/         # BERT模型缓存
├── main.py            # 主程序
├── model.py           # 模型定义
├── data_loader.py     # 数据加载
└── requirements.txt   # 依赖列表
```

## 注意事项

1. 首次运行时会自动下载所需的数据集和BERT模型
2. 训练过程中会自动保存检查点
3. 支持断点续训
4. CPU训练可能较慢，建议使用GPU