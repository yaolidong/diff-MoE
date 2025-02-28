# Diff-MoE: 多模态混合专家模型

这是一个基于PyTorch实现的多模态混合专家模型，支持图像分类任务。模型采用混合专家(Mixture of Experts)结构，能够更有效地处理不同类型的输入数据。

## 环境要求

- Python 3.8+
- PyTorch 2.0.0+
- CUDA (可选，用于GPU加速)
- torchvision
- matplotlib, seaborn (用于可视化)
- scikit-learn (用于评估)
- transformers (BERT模型)

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

## 项目结构与文件说明

```
diff-MoE/
├── main.py               # 主函数，程序入口
├── train.py              # 训练相关代码
├── model.py              # 多模态模型设计
├── AttentiveRouter.py    # 路由策略实现
├── export.py             # 专家层设计
├── data_loader.py        # 数据集加载
├── label_to_text.py      # 数据集标签映射
├── test_utils.py         # 验证模型性能工具
├── visualization.py      # 可视化功能
├── config.py             # 配置参数管理
├── data/                 # 数据集目录
├── visualizations/       # 可视化结果保存目录
├── checkpoints/          # 模型检查点目录
└── requirements.txt      # 依赖库清单
```

### 文件职责说明

1. **main.py**: 程序入口，包含主函数和环境配置
2. **train.py**: 负责模型训练和验证的所有代码，包括训练循环、优化器设置、学习率调度等
3. **model.py**: 定义多模态混合专家模型的架构和前向传播过程
4. **AttentiveRouter.py**: 实现专家路由策略，决定将输入分配给哪些专家处理
5. **export.py**: 定义专家层结构，封装单个专家的计算逻辑
6. **data_loader.py**: 负责数据集加载、预处理和批处理
7. **label_to_text.py**: 提供数据集标签到文本描述的映射
8. **test_utils.py**: 包含测试和评估模型性能的工具函数
9. **visualization.py**: 提供各种可视化功能，如预测结果、专家分配、注意力权重等
10. **config.py**: 集中管理所有配置参数，支持从环境变量和配置文件加载

## 主要特性

- **多模态混合专家路由**: 模型使用多个专家处理不同类型的输入特征
- **灵活的配置系统**: 支持环境变量和配置文件两种方式配置参数
- **混合精度训练**: 支持FP16混合精度训练以加速计算
- **梯度检查点**: 支持梯度检查点以减少显存占用
- **多种可视化**: 支持预测结果、专家分配、注意力权重等多种可视化
- **兼容多种数据集**: 支持CIFAR10、Fashion-MNIST等多种数据集

## 高级使用

### 自定义配置

可以通过环境变量或配置文件修改训练参数：

```bash
# 通过环境变量
export TRAINING_learning_rate=0.001
export TRAINING_batch_size=64
python main.py

# 通过配置文件
# 在configs/目录下创建配置文件，然后使用--config参数
python main.py --config configs/my_config.yaml
```

### 添加新数据集

要添加新的数据集，需要：
1. 在config.py中添加新的数据集配置类
2. 在label_to_text.py中添加标签映射函数
3. 在data_loader.py中更新get_dataset_and_loaders函数

## 常见问题

- **训练太慢**: 尝试使用GPU，或减小batch_size和模型大小
- **显存不足**: 启用梯度检查点，减小batch_size，或使用梯度累积
- **可视化问题**: 确保matplotlib和seaborn正确安装
- **数据加载错误**: 检查数据目录和权限设置

## 引用

如果您使用了本项目，请引用：

```
@misc{diff-moe2023,
  author = {Your Name},
  title = {Diff-MoE: Multi-modal Mixture of Experts Model},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/yourusername/diff-MoE}
}
```