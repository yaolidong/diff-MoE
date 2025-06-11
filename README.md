# Diff-MoE: 多模态混合专家模型

这个项目实现了一个基于混合专家(Mixture of Experts, MoE)的多模态神经网络模型，用于图像分类任务，支持CIFAR10、Fashion-MNIST和Flickr8k数据集。

## 项目结构

```
diff-MoE/
├── .gitignore            # Git忽略文件配置
├── config.py             # 配置参数
├── configs/              # 配置文件目录
│   ├── model.yaml        # 模型配置
│   ├── training.yaml     # 训练配置
│   ├── cifar10.yaml      # CIFAR10数据集配置
│   ├── fashion_mnist.yaml # Fashion-MNIST数据集配置
│   └── visualization.yaml # 可视化配置
├── data_loader.py        # 数据加载器
├── data_utils.py         # 数据集下载和预处理工具
├── datasets.py           # 数据集定义和标签文本描述
├── diff-MoE.code-workspace # VS Code 工作区配置文件
├── image_captioning.py   # 图像字幕生成脚本
├── image_classification.py # 图像分类脚本
├── main.py               # 主程序入口
├── model.py              # 模型定义
├── profile_analyzer.py   # 性能分析工具
├── README.md             # 项目说明
├── requirements.txt      # 依赖库
├── test.py               # 测试和评估函数
├── train.py              # 训练函数
├── training_example.py   # 训练示例脚本
└── utils.py              # 工具函数和可视化
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

Diff-MoE是一个基于Transformer架构的多模态混合专家模型。其核心组件是`UnifiedModalEncoder`，一个包含多种类型专家的混合专家 (MoE) 模块。这些模块被用于构建图像编码器、文本编码器以及跨模态融合层。主要特点包括：

1. **混合专家 (MoE) 模块 (`UnifiedModalEncoder`)**: 每个模块内包含多种专家：
    - 一个全局共享专家 (`global_expert`)，处理所有输入token。
    - 模态特定专家 (`vision_expert`, `text_expert`)，分别处理图像和文本token（在非融合层中）。
    - 一个通用专家池 (`general_experts`)，由`AttentiveRouter`为每个token选择一部分专家进行处理。
2. **多模态融合**: 通过`CrossModalFusion`层结合图像和文本特征，该层也利用了`UnifiedModalEncoder`模块。
3. **Attention机制**: 在`UnifiedModalEncoder`中使用自注意力和交叉注意力（在`CrossModalFusion`中）进行特征交互。
4. **分层编码**: `ImageEncoder`和`TextEncoder`分别由多个`UnifiedModalEncoder`堆叠而成，用于深度特征提取。

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
|  |  +------------------------------------------+ |
|  |  |       混合专家模块 (UnifiedModalEncoder)     | |
|  |  |  (包含全局专家, 模态特定专家, 通用专家池)  | |
|  |  +------------------------------------------+ |
|  |  | +------------------+  +-----------------+ | |
|  |  | | 全局/模态特定专家 |  |  通用专家池      | | |
|  |  | +------------------+  +-----------------+ | |
|  |  |                       | +-------------+ | | |
|  |  |                       | | 注意力路由器 | | | |
|  |  |                       | +-------------+ | | |
|  |  +------------------------------------------+ |
|  |                         |                      |
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

1.  **`Expert` (专家模块)**:
    *   基础的前馈网络单元，由两个全连接层、激活函数 (如GELU, ReLU, SiLU)、Layer Normalization和Dropout组成。
    *   用于构建`UnifiedModalEncoder`中的各种专家。

2.  **`PatchEmbed` (图像分块嵌入层)**:
    *   将输入图像分割成固定大小的块 (patches)，并通过卷积层将每个块线性嵌入为向量。
    *   为图像数据进入Transformer结构做准备。

3.  **`AttentiveRouter` (注意力路由器)**:
    *   在`UnifiedModalEncoder`内部使用，基于注意力机制为每个输入token从`general_experts`（通用专家池）中选择top-k个专家。
    *   包含正交初始化和可选的噪声辅助路由。

4.  **`UnifiedModalEncoder` (统一模态编码器/混合专家模块)**:
    *   模型的核心MoE构建块，包含：
        *   **自注意力层**: 捕捉输入序列内不同token间的关系。
        *   **多种专家类型**:
            *   `global_expert`: 一个全局共享专家，处理所有token。
            *   `vision_expert` 和 `text_expert`: 模态特定专家，分别处理图像和文本token（在专门的图像/文本编码层中，非融合层）。
            *   `general_experts`: 一个通用专家池，由`AttentiveRouter`动态选择。
        *   **残差连接和Layer Normalization**: 保证训练稳定性。
    *   支持梯度检查点以节省显存。
    *   *注意: ASCII图中的“共享专家”指`global_expert`，“模态特定专家”指`vision_expert`/`text_expert`。通用专家池由路由器管理。*

5.  **`ImageEncoder` (图像编码器)**:
    *   由多个`UnifiedModalEncoder`模块堆叠而成，专门用于处理图像patch嵌入序列，提取深层图像特征。

6.  **`TextEncoder` (文本编码器)**:
    *   与`ImageEncoder`类似，由多个`UnifiedModalEncoder`模块堆叠而成，处理文本token嵌入序列，提取深层文本特征。

7.  **`CrossModalFusion` (跨模态融合层)**:
    *   负责融合来自`ImageEncoder`和`TextEncoder`的特征。
    *   通常包含交叉注意力机制，允许图像和文本特征相互作用。
    *   也可能使用`UnifiedModalEncoder`模块进行深度融合处理。

8.  **`MultiModalMoE` (多模态混合专家主模型)**:
    *   顶层模型，整合了上述所有组件（`PatchEmbed`, `ImageEncoder`, `TextEncoder`, `CrossModalFusion`）。
    *   处理图像和（可选的）文本输入，最终输出融合后的特征表示。
    *   管理位置编码、模态类型嵌入等。

*之前的“图像处理”和“文本处理”小节内容已整合到上述组件描述中。*

### 模型配置参数

下表列出了模型的主要配置参数及其默认值。这些值通常在 `configs/model.yaml` 中定义或在模型初始化时设置。

| 参数                             | 默认值    | 描述                                                                 |
| -------------------------------- | --------- | -------------------------------------------------------------------- |
| `img_size`                       | 28        | 输入图像大小                                                           |
| `patch_size`                     | 4         | 图像patch大小                                                          |
| `in_channels`                    | 1         | 输入图像通道数                                                         |
| `embed_dim`                      | 512       | 嵌入维度                                                               |
| `num_general_experts`            | 4         | 通用专家池中的专家数量 (`UnifiedModalEncoder`中的`general_experts`)      |
| `num_modality_specific_experts`  | 2         | 模态特定专家数量 (`vision_expert` 和 `text_expert`，每个`UnifiedModalEncoder`中) |
| `top_k`                          | 2         | 每个token为通用专家池选择的专家数量                                     |
| `num_heads`                      | 8         | 注意力头数量                                                             |
| `img_encoder_layers`             | 6         | 图像编码器中的`UnifiedModalEncoder`层数                               |
| `text_encoder_layers`            | 4         | 文本编码器中的`UnifiedModalEncoder`层数                               |
| `fusion_layers`                  | 3         | 跨模态融合层中的`UnifiedModalEncoder`层数                             |
| `num_classes`                    | 10        | 输出类别数量 (通常由下游分类头定义)                                       |
| `dropout`                        | 0.2       | Dropout比率                                                            |
| `activation`                     | 'gelu'    | 激活函数类型 (如 'gelu', 'relu', 'silu')                               |
| `layer_norm_eps`                 | 1e-5      | Layer Normalization中的epsilon值                                       |
| `initializer_range`              | 0.02      | 权重初始化范围                                                           |
| `use_gradient_checkpointing`     | `true`    | 是否使用梯度检查点以节省显存                                                 |
| `vocab_size`                     | 50000     | 文本词汇表大小                                                           |
| `max_text_len`                   | 32        | 最大文本序列长度                                                         |
| `text_embed_dim`                 | 128       | 初始文本嵌入维度 (在投影到`embed_dim`之前)                               |

## 损失函数

在训练Diff-MoE模型时，总的损失函数由两大部分构成：**模型内部辅助损失** (由`MultiModalMoE`模型自身计算和返回) 和 **任务特定损失** (在训练脚本中根据具体任务定义，例如分类任务的交叉熵损失)。

### 1. 模型内部辅助损失 (Model-Internal Auxiliary Losses)

这些损失由`MultiModalMoE`模型在其`forward`方法中计算，旨在帮助模型学习更鲁棒和均衡的表示，并优化混合专家 (MoE) 的路由机制。模型返回的字典中包含一个键为 `'router_loss'` 的项，这是以下所有加权辅助损失的总和。

各辅助损失组件说明如下：

*   **路由器 Z-Loss (Router Z-Loss / 正则化损失)**:
    *   **目的**: 对路由器的门控 logits 进行正则化，鼓励其输出的概率分布不过于集中。
    *   **计算**: 由模型内部的 `compute_z_loss` 方法计算。其计算方式为：对每个路由器的输出`logits`应用`softmax`得到概率，计算每个专家在批次和序列维度上的平均激活概率，然后取这些平均概率的平方的均值，并乘以专家总数。
    *   **权重**: `router_z_loss_weight` (默认为 `0.001`)。

*   **路由器负载均衡损失 (Router Load Balancing Loss)**:
    *   **目的**: 鼓励模型将输入 token 均匀地分配给通用专家池 (`general_experts`) 中的各个专家，避免部分专家过载而其他专家空闲。
    *   **计算**: 基于KL散度，比较实际的专家token分配比例与理想的均匀分配比例。由模型内部的 `compute_load_loss` 方法计算。
        ```
        负载目标分布 = 均匀分布(1 / num_general_experts)
        专家利用率 = 每个通用专家处理的token数 / 总token数
        负载均衡损失 = KL散度(专家利用率, 负载目标分布)
        ```
    *   **权重**: `router_balance_loss_weight` (默认为 `0.01`)。

*   **跨模态对齐损失 (Cross-Modal Alignment Loss)**:
    *   **目的**: (当存在文本输入时) 促使模型学习图像和对应文本的相似或对齐的表示。这是通过最大化图像和文本特征的平均池化表示之间的余弦相似度来实现的。
    *   **计算**: `-torch.sum(F.normalize(img_features_mean) * F.normalize(text_features_mean))`，然后进行批次平均。
    *   **权重**: `cross_modal_alignment_weight` (默认为 `0.1`)。

*   **对比损失 (Contrastive Loss)**:
    *   **目的**: (当存在文本输入时) 学习一个共享的嵌入空间，其中匹配的图像-文本对的嵌入被拉近，而不匹配的对被推远。
    *   **计算**: 通常使用InfoNCE损失变体，基于图像到文本和文本到图像的相似度矩阵计算交叉熵损失。
    *   **权重**: `contrastive_loss_weight` (默认为 `0.1`)。

**模型内部组合损失计算**:
`MultiModalMoE`模型将上述加权损失汇总为一项（在返回字典中键为`'router_loss'`）：
```
Combined_Model_Internal_Loss = router_z_loss_weight * total_router_z_loss +
                             router_balance_loss_weight * total_router_balance_loss +
                             cross_modal_alignment_weight * total_cross_modal_loss +
                             contrastive_loss_weight * total_contrastive_loss
```
*(注意：`total_router_z_loss`等是在模型内部对来自不同编码器层或处理流程的相应损失进行累加的结果。)*

### 2. 任务特定损失 (Task-Specific Loss)

该损失取决于模型的具体应用场景，在训练脚本中定义和计算。例如：

*   **分类损失 (Classification Loss)**:
    *   如果模型用于图像分类或多模态分类任务，通常会在`MultiModalMoE`模型输出的特征嵌入之上添加一个分类头 (如一个线性层)。
    *   然后使用标准交叉熵损失 (CrossEntropyLoss) 计算预测 logits 与真实标签之间的差异。
    ```
    分类损失 = CrossEntropyLoss(classification_head_logits, labels)
    ```

### 3. 总训练损失 (Overall Training Loss)

最终用于反向传播的总训练损失是模型内部辅助损失与任务特定损失的和：
```
总训练损失 = Combined_Model_Internal_Loss + Task_Specific_Loss
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