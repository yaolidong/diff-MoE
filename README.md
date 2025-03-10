# Diff-MoE: 多模态混合专家模型

这个项目实现了一个基于混合专家(Mixture of Experts, MoE)的多模态神经网络模型，用于图像分类任务，支持CIFAR10、Fashion-MNIST和Flickr8k数据集。

## 项目结构

```
diff-MoE/
├── main.py               # 主程序入口
├── model.py              # 模型定义
├── train.py              # 训练函数
├── test.py               # 测试和评估函数
├── utils.py              # 工具函数和可视化
├── data_loader.py        # 数据加载器
├── datasets.py           # 数据集定义和标签文本描述
├── data_utils.py         # 数据集下载和预处理工具
├── config.py             # 配置参数
├── configs/              # 配置文件目录
│   ├── model.yaml        # 模型配置
│   ├── training.yaml     # 训练配置
│   ├── cifar10.yaml      # CIFAR10数据集配置
│   ├── fashion_mnist.yaml # Fashion-MNIST数据集配置
│   └── visualization.yaml # 可视化配置
├── requirements.txt      # 依赖库
├── README.md             # 项目说明
└── README_FLICKR8K.md    # Flickr8k数据集使用说明
```

## 功能模块

- **数据处理**：
  - `data_loader.py`: 提供数据加载函数和批处理
  - `datasets.py`: 定义自定义数据集类及标签到文本描述的转换
  - `data_utils.py`: 数据集下载和预处理工具

- **模型**：
  - `model.py`: 实现MultiModalMoE模型和各个组件
  - `config.py`: 模型和训练基础配置参数
  - `configs/`: 包含各数据集的具体配置YAML文件

- **训练与测试**：
  - `train.py`: 提供训练循环和模型优化
  - `test.py`: 模型评估和测试功能

- **工具与可视化**：
  - `utils.py`: 工具函数和可视化功能
  - `main.py`: 命令行接口和程序入口

- **缓存目录**：
  - `clip_cache/`: CLIP模型和tokenizer缓存
  - `bert_cache/`: BERT模型和tokenizer缓存（旧版本使用）

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
# 训练CIFAR10模型
python main.py --dataset cifar10 --mode train --batch_size 64 --epochs 10

# 训练Fashion-MNIST模型
python main.py --dataset fashion_mnist --mode train --batch_size 64 --epochs 10

# 训练Flickr8k模型
python main.py --dataset flickr8k --mode train --batch_size 32 --epochs 20
```

### 测试模型

```bash
# 测试CIFAR10模型
python main.py --dataset cifar10 --mode test --checkpoint checkpoints/cifar10_model.pth

# 测试Fashion-MNIST模型
python main.py --dataset fashion_mnist --mode test --checkpoint checkpoints/fashion_mnist_model.pth

# 测试Flickr8k模型
python main.py --dataset flickr8k --mode test --checkpoint checkpoints/flickr8k_model.pth
```

### 训练并测试

```bash
# 训练并测试CIFAR10模型
python main.py --dataset cifar10 --mode train_test --batch_size 64 --epochs 10
```

## 命令行参数

- `--dataset`: 数据集名称，选项：cifar10, fashion_mnist, flickr8k
- `--mode`: 运行模式，选项：train, test, train_test
- `--batch_size`: 批次大小
- `--epochs`: 训练轮次数
- `--lr`: 学习率
- `--weight_decay`: 权重衰减
- `--checkpoint`: 模型检查点路径
- `--save_dir`: 保存目录

## 模型架构

Diff-MoE是一个基于Transformer架构的多模态混合专家模型，主要特点：

1. **混合专家层**: 包含共享和模态特定的专家网络，通过路由器分配不同的输入到不同的专家
2. **多模态融合**: 结合图像和文本特征进行预测
3. **Attention机制**: 使用自注意力和交叉注意力进行特征交互

### 模型架构详细说明

```
+------------------------------------------+
|             MultiModalMoE                |
+------------------------------------------+
|                                          |
|  +------------+       +---------------+  |
|  |   图像     |       |     文本      |  |
|  |   编码器   |       |     编码器    |  |
|  +-----+------+       +-------+-------+  |
|        |                      |          |
|        v                      v          |
|  +------------+       +---------------+  |
|  |   图像     |       |   CLIP文本    |  |
|  |   嵌入层   |       |   Tokenizer   |  |
|  +-----+------+       +-------+-------+  |
|        |                      |          |
|        v                      v          |
|  +----------------------------------------+
|  |         统一模态编码器 (UnifiedModalEncoder) |
|  +----------------------------------------+
|  |                                        |
|  |  +-------------+    +---------------+  |
|  |  | 自注意力机制 |    | 交叉注意力机制 |  |
|  |  +------+------+    +-------+-------+  |
|  |         |                   |          |
|  |         v                   v          |
|  |  +------------------+----------------+ |
|  |  |          混合专家层 (MoE)          | |
|  |  +----------------------------------+ |
|  |  |                                  | |
|  |  | +------------+ +---------------+ | |
|  |  | | 共享专家   | | 模态特定专家  | | |
|  |  | +------------+ +---------------+ | |
|  |  |                                  | |
|  |  | +------------+                   | |
|  |  | | 注意力路由器|                   | |
|  |  | +------------+                   | |
|  |  +----------------------------------+ |
|  |                  |                    |
|  +------------------+--------------------+
|                     |                     
|                     v                     
|  +------------------+--------------------+
|  |               分类头                   |
|  +------------------+--------------------+
|                     |                     
+---------------------|---------------------+
                      v                     
                    输出                    
```

#### 关键组件

1. **专家模块 (Expert)**: 
   - 每个专家由两个全连接层组成，带有激活函数和Layer Normalization
   - 支持的激活函数: GELU, ReLU, SiLU
   - 包含Dropout机制以防止过拟合

2. **注意力路由器 (AttentiveRouter)**:
   - 基于注意力机制的路由算法
   - 为每个输入token分配top-k个专家
   - 使用正交初始化提高稳定性
   - 支持噪声辅助路由以增强训练过程中的探索

3. **统一模态编码器 (UnifiedModalEncoder)**:
   - 多头自注意力机制
   - 跨模态交叉注意力融合
   - 混合专家前馈网络
   - 支持梯度检查点以节省显存

4. **图像处理**:
   - Patch嵌入层将图像分割为固定大小的块
   - 位置编码提供空间信息

5. **文本处理**:
   - 使用嵌入层将文本token转化为向量
   - 支持自注意力处理序列信息
   - 使用CLIP tokenizer进行文本分词，最大序列长度为77

### 模型配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| img_size | 28 | 输入图像大小 |
| patch_size | 4 | 图像patch大小 |
| in_channels | 1 | 输入图像通道数 |
| embed_dim | 512 | 嵌入维度 |
| num_shared_experts | 4 | 共享专家数量 |
| num_modality_specific_experts | 2 | 模态特定专家数量 |
| top_k | 2 | 每个输入选择的专家数量 |
| num_heads | 8 | 注意力头数量 |
| num_layers | 6 | Transformer层数 |
| num_classes | 10 | 分类类别数量 |
| dropout | 0.1 | Dropout比率 |
| activation | 'gelu' | 激活函数类型 |

## 损失函数

Diff-MoE模型的总损失函数由两部分组成：

1. **分类损失**：使用标准交叉熵损失(CrossEntropyLoss)计算预测与真实标签之间的差异
   ```
   分类损失 = CrossEntropyLoss(logits, labels)
   ```

2. **路由损失**：基于KL散度计算的专家负载均衡损失，确保专家被均匀使用
   ```
   负载目标分布 = 均匀分布(1/num_experts)
   专家利用率 = 每个专家处理的token数 / 总容量
   路由损失 = KL散度(专家利用率, 负载目标分布)
   ```

3. **总损失**：
   ```
   总损失 = 分类损失 + 路由损失
   ```

### 优化器

- **默认优化器**: AdamW
- **学习率**: 0.0005
- **权重衰减**: 0.01
- **梯度累积**: 支持梯度累积减少显存占用

### 训练技巧

1. **梯度累积**：每个batch的梯度会累积几个步骤后再更新参数，有效增大batch size
2. **梯度检查点**：可选使用梯度检查点技术以节省显存
3. **学习率调度**：支持学习率调度器动态调整学习率

## 推理过程

推理时，模型接收图像和可选的文本输入：

1. 图像通过Patch嵌入层处理后得到图像特征
2. 文本(如果有)通过嵌入层处理后得到文本特征
3. 两种特征通过统一模态编码器处理，包括：
   - 自注意力机制捕获同一模态内的关系
   - 交叉注意力机制融合不同模态的信息
   - 通过注意力路由器将特征分配给不同专家
   - 聚合专家输出得到最终表示
4. 最后通过分类头输出类别概率

## 可视化与解释性

模型支持以下可视化功能：
1. **路由决策可视化**：展示每个专家处理的token数量和分布
2. **注意力图**：展示模型在不同输入部分的关注程度
3. **专家利用率**：分析不同专家的使用情况和负载均衡
4. **多模态交互**：展示图像和文本特征如何相互影响

## 支持的数据集

- **CIFAR10**: 10类通用对象图像分类
- **Fashion-MNIST**: 10类服装图像分类
- **Flickr8k**: 图像-文本对数据集，用于多模态学习

## 可视化示例

训练过程中会生成以下可视化：
- 训练曲线（损失和准确率）
- 预测结果示例
- 混淆矩阵
- 专家分配可视化

可视化结果保存在`visualizations/`目录下。

## 技术特点

本项目结合了多种先进技术：

1. **混合专家模型**：通过路由机制实现高效计算资源分配
2. **多模态融合**：同时处理图像和文本信息
3. **CLIP文本编码**：使用OpenAI的CLIP-BPE tokenizer处理文本，相比BERT在多模态任务上表现更好
4. **注意力机制**：使用自注意力和交叉注意力进行深度特征提取
5. **可视化分析**：支持模型决策过程的可视化

## 环境要求

本项目需要以下依赖：

```
torch>=1.10.0        # PyTorch深度学习框架
torchvision>=0.11.0  # PyTorch计算机视觉工具
transformers>=4.18.0 # Hugging Face Transformers库，用于CLIP和BERT
matplotlib>=3.5.0    # 数据可视化
numpy>=1.21.0        # 数值计算
tqdm>=4.62.0         # 进度条显示
scikit-learn>=1.0.0  # 机器学习工具
seaborn>=0.11.0      # 统计数据可视化
pyyaml>=6.0.0        # YAML配置文件处理
pillow>=9.0.0        # 图像处理
scipy>=1.7.0         # 科学计算
```

对于Flickr8k数据集，还需要额外安装：
```
requests              # HTTP请求库，用于下载数据集
```

硬件要求：
- 建议使用至少8GB显存的GPU进行训练
- 对于Flickr8k数据集，建议使用至少16GB显存的GPU