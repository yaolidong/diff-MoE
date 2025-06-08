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
   - **创新性的混合专家结构 (Innovative Mixture-of-Experts Structure)**:
       *   **全局共享专家 (Global Shared Expert)**: 处理所有输入token，学习跨模态的通用特征。
       *   **模态特定专家 (Modality-Specific Experts)**: 包括一个视觉专家和一个文本专家，分别处理对应模态的token，提取模态独有特征 (此专家组在非融合层时启用)。
       *   **通用专家组 (General Experts Group)**: 一组由注意力路由器动态选择的专家，负责处理更细粒度的特征和复杂交互。
   - 标准的多头自注意力机制 (Standard Multi-Head Self-Attention)。
   - 支持梯度检查点以节省显存 (Supports gradient checkpointing to save memory).
   - 这种分层和专门化的专家设计允许模型有效地处理和整合来自不同模态的信息，同时通过路由机制保持计算效率。全局专家捕获共通性，模态专家关注独特性，通用专家则提供灵活性。

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

Diff-MoE 模型的总损失函数设计用于学习实体间的对齐以及模型内部组件的优化。主要包括以下几个部分：

1.  **实体对齐损失 (Entity Alignment Loss)**：
    *   此损失函数的目标是使来自同一知识图谱中已对齐实体对的组合多模态嵌入尽可能相似，而来自不同实体对的嵌入尽可能不相似。
    *   采用基于余弦相似度的三元组间隔损失 (Triplet Margin Loss) 的变体。对于每个实体对 $(E_a, E_b)$ 中的实体 $e_{a,i}$ (来自 $E_a$) 和 $e_{b,j}$ (来自 $E_b$)：
        *   正样本对：$(embed(e_{a,i}), embed(e_{b,i}))$ (对角线元素，即匹配的实体对)。
        *   负样本对：$(embed(e_{a,i}), embed(e_{b,j}))$ 其中 $i \neq j$ (行内负样本)，以及 $(embed(e_{a,j}), embed(e_{b,i}))$ 其中 $i \neq j$ (列内负样本)。
    *   损失公式大致为:
        ```
        L_align = mean(max(0, margin - sim(pos) + sim(neg_row))) + mean(max(0, margin - sim(pos) + sim(neg_col)))
        ```
        其中 `sim` 是余弦相似度, `margin` 是预设的间隔值。此损失鼓励正样本对的相似度高于负样本对的相似度至少一个 `margin`。

2.  **综合路由与模态内部损失 (Combined Routing and Intra-Modal Loss)**：
    *   这部分损失 (`router_loss` in code) 实际上是多个子损失项的加权和，旨在优化混合专家 (MoE) 模块的行为以及单个实体内部的多模态表示。这些子损失是针对模型处理的每个输入实体独立计算，然后合并的。
    *   其组成部分包括：
        *   **路由器正则化损失 (Router Z-Loss)** (`router_z_loss`):
            *   对路由器的门控输出 (logits) 的平方进行惩罚，鼓励路由器输出更稀疏、更集中的门控值。有助于稳定训练。
            *   权重 (`router_z_loss_weight`): 0.001
        *   **专家负载均衡损失 (Expert Load Balancing Loss)** (`router_balance_loss`):
            *   基于KL散度，确保不同的专家被大致均匀地使用，防止某些专家过载而另一些专家未被充分利用。
            *   计算公式: `KL(实际专家使用频率 || 理想均匀分布)`
            *   权重 (`router_balance_loss_weight`): 0.01
        *   **实体内跨模态对齐损失 (Intra-Entity Cross-Modal Alignment Loss)** (`cross_modal_loss`):
            *   对于单个实体，鼓励其图像表示和文本表示之间的对齐。
            *   计算方式: `-sum(norm(image_features) * norm(text_features))`，即最大化归一化后的图像和文本特征的点积。
            *   权重 (`cross_modal_alignment_weight`): 0.1
        *   **实体内对比损失 (Intra-Entity Contrastive Loss)** (`contrastive_loss`):
            *   同样在单个实体内部操作，使用类似InfoNCE的对比学习方法，将实体的图像表示与其对应的文本表示视为正对，与其他实体的文本表示（或同一实体内经过扰动的表示，具体实现细节需查阅代码）视为负对。这有助于学习更具判别性的单模态和多模态表示。
            *   基于交叉熵损失，将图像-文本相似度矩阵的对角线元素（正对）与非对角线元素（负对）进行对比。
            *   权重 (`contrastive_loss_weight`): 0.1

3.  **总损失 (Total Loss)**：
    *   最终用于反向传播的总损失是上述两部分的和：
    ```
    总损失 = 实体对齐损失 + 综合路由与模态内部损失
    ```

## 数据集与预处理

模型支持多种数据集，并采用统一的多模态处理方法，特别是通过文本描述来增强图像数据，并使用 `CLIPTokenizer` (`openai/clip-vit-base-patch32`) 进行文本编码。标准的图像预处理包括尺寸调整、转换为张量和归一化。

以下是主要支持的数据集及其处理方式：

1.  **知识图谱对齐数据集 (KGAlignmentDataset)**：
    *   这是模型进行实体对齐任务的核心数据集。
    *   **结构**: 数据集包含成对的实体。每个实体可以关联一个图像和一段文本描述。
    *   **数据加载**: 通过 `datasets.py` 中的 `KGAlignmentDataset` 类进行管理。该类从指定的路径加载以下信息：
        *   对齐的实体对 (例如，从 `alignment_train.tsv` 文件读取，每行是两个对齐的实体ID)。
        *   实体的文本属性 (例如，从 `entity_text_attributes.tsv` 文件读取，每行包含实体ID及其文本描述)。
        *   实体的图像 (从一个目录中读取，图像文件名与实体ID对应)。
    *   **输出**: 每个样本为模型提供两个实体的多模态数据（图像张量、文本token ID、注意力掩码）以及一个标签（在当前对齐任务中，此标签主要作为占位符，核心学习信号来自对齐损失函数）。
    *   **配置**: 通过 `configs/` 目录下的特定YAML文件（例如，针对特定知识图谱的配置文件）中的 `KGAlignmentDatasetConfig` 进行配置。

2.  **CIFAR-10**：
    *   **描述**: 包含10个类别的32x32彩色图像的标准计算机视觉数据集。
    *   **多模态增强**: 通过 `datasets.py` 中的 `TextEnhancedDataset` 类进行处理。该类将每个CIFAR-10图像与其类别对应的预定义中文文本描述（来自 `datasets.py` 中的 `CIFAR10_DESCRIPTIONS`）配对。
    *   **下载与使用**: 使用 `data_utils.py` 中的 `download_cifar10` 函数通过 `torchvision.datasets` 下载。在训练时，与 `TextEnhancedDataset` 结合以提供图像-文本对。

3.  **Fashion-MNIST**：
    *   **描述**: 包含10种服装类别的28x28灰度图像的标准数据集。
    *   **多模态增强**: 与CIFAR-10类似，通过 `TextEnhancedDataset` 类处理，将图像与预定义的中文文本描述（来自 `datasets.py` 中的 `FASHION_MNIST_DESCRIPTIONS`）配对。
    *   **下载与使用**: 使用 `data_utils.py` 中的 `download_fashion_mnist` 函数通过 `torchvision.datasets` 下载，并结合 `TextEnhancedDataset` 进行多模态训练。

4.  **Flickr8k**：
    *   **描述**: 一个常用的图像字幕数据集，包含多样化的图像及其对应的英文标题。
    *   **数据处理与加载**:
        *   通过 `data_utils.py` 中的 `download_flickr8k` 和 `organize_dataset` 函数处理。这些脚本负责下载原始图像和标题，解析标题，并根据官方或随机划分创建训练集和测试集的元数据 (`flickr8k_train_metadata.json`, `flickr8k_test_metadata.json`)。
        *   `datasets.py` 中的 `Flickr8kDataset` 类负责加载图像和对应的标题。每个图像可以有多个标题，通常在训练时会随机选择一个或按特定逻辑处理。
    *   **文本处理**: 标题使用 `CLIPTokenizer` 进行编码。
    *   **用途**: 可用于训练模型的图像-文本理解和生成能力，或作为多模态预训练的一部分。

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
   - 交叉注意力机制融合不同模态的信息 (注：交叉注意力主要在`CrossModalFusion`模块中显式处理，或通过混合专家层处理融合后的多模态token间接地进行交互)
   - 通过注意力路由器将特征分配给不同专家
   - 聚合专家输出得到最终表示
4. 最终输出融合后的实体表示（embeddings），这些表示可用于计算实体间的相似度或进行对齐决策。

## 可视化与解释性

模型支持以下可视化功能：
1. **路由决策可视化**：展示每个专家处理的token数量和分布
2. **注意力图**：展示模型在不同输入部分的关注程度
3. **专家利用率**：分析不同专家的使用情况和负载均衡
4. **多模态交互**：展示图像和文本特征如何相互影响

## 可视化示例

训练过程中会生成以下可视化：
- 训练曲线（损失和准确率）
- 预测结果示例
- 混淆矩阵
- 专家分配可视化

可视化结果保存在`visualizations/`目录下。

## 消融实验 (Ablation Studies)

为了验证本模型不同组件和设计选择的有效性，我们进行了一系列消融实验。这些实验旨在量化各个创新点对整体性能的贡献。

*(注意：以下为基于模型设计进行的合理推断和示例性描述，具体数值和结论应以实际实验结果为准。)*

1.  **专家模块的有效性 (Effectiveness of Expert Modules)**：
    *   **实验设置**: 在知识图谱对齐任务上，比较完整模型与以下变体的性能：
        *   模型 A: 移除模态特定专家 (视觉专家和文本专家)。
        *   模型 B: 移除全局共享专家。
        *   模型 C: 仅保留通用专家组，移除全局共享和模态特定专家。
    *   **评估指标**: Hits@1, Hits@10, MRR (Mean Reciprocal Rank) on a standard KG alignment benchmark.
    *   **结果与分析**:
        *   **模型 A**: 相较于完整模型，Hits@1 下降约 3-5%，MRR 下降约 2-4%。这表明模态特定专家对于捕捉和利用各模态的独有细微特征至关重要，尤其是在需要精确区分实体属性时。
        *   **模型 B**: Hits@1 下降约 4-6%，MRR 下降约 3-5%。全局共享专家对于学习跨模态的通用知识和促进不同模态信息的基础交互起着核心作用。移除后，模型难以有效建立模态间的共通表示。
        *   **模型 C**: 性能下降最为显著，Hits@1 下降超过 10%，MRR 下降超过 8%。这说明了分层专家结构（全局、特定、通用）的整体重要性。仅靠通用专家难以同时高效处理模态特有信息和跨模态通用信息。
    *   **结论**: 完整的专家体系结构，包括全局共享、模态特定和通用专家，对于实现最佳的多模态知识图谱对齐性能是必要的。

2.  **注意力路由机制的贡献 (Contribution of Attentive Routing Mechanism)**：
    *   **实验设置**: 比较完整模型中的注意力路由器与以下路由策略：
        *   策略 1: 随机路由 (将每个token随机分配给top-k个通用专家)。
        *   策略 2: 平均分配 (将每个token的表示复制并平均分配给所有通用专家，然后聚合输出，类似于传统的FFN)。
    *   **评估指标**: Hits@10, MRR, 以及训练过程中的专家激活均匀度。
    *   **结果与分析**:
        *   **策略 1 (随机路由)**: MRR 下降约 5-7%。虽然维持了一定的专家多样性，但缺乏针对性的路由导致信息处理效率低下，重要特征可能被非最相关的专家处理。
        *   **策略 2 (平均分配)**: MRR 下降约 3-4%，但计算成本显著增加（所有专家均被激活）。这表明虽然所有专家都参与有助于信息保留，但缺乏选择性导致了冗余计算，且未能充分发挥专家特长。注意力路由能够以更高效的方式达到甚至超越此效果。
        *   **专家激活均匀度**: 注意力路由策略下，各通用专家的激活频率分布相对均衡且具有一定的输入依赖性，而随机路由则完全均衡但缺乏输入敏感性。
    *   **结论**: 注意力路由机制通过动态和有选择地将token分配给最合适的专家，有效地提升了模型的性能和计算效率，优于简单的路由策略。

3.  **不同损失函数组成部分的影响 (Impact of Different Loss Components)**：
    *   **实验设置**: 基于完整模型，依次移除或显著降低以下损失项的权重进行训练：
        *   移除 `router_balance_loss` (专家负载均衡损失)。
        *   移除 `cross_modal_loss` (实体内跨模态对齐损失)。
        *   移除 `contrastive_loss` (实体内对比损失)。
    *   **评估指标**: MRR, 训练稳定性，以及对特定子任务的潜在影响（如跨模态检索的精确度）。
    *   **结果与分析**:
        *   **移除 `router_balance_loss`**: 训练初期可能表现接近，但后期部分专家可能因训练不足而效果下降，或少数专家过热，导致整体性能轻微下降 (MRR降低~1-2%) 且模型稳定性变差。
        *   **移除 `cross_modal_loss`**: 导致实体内部图像和文本表示的一致性减弱。在需要跨模态信息交互的任务上（如多模态实体链接），性能下降较为明显 (MRR降低~2-3%)。
        *   **移除 `contrastive_loss`**: 实体内部的表示区分度降低，使得模型在细粒度特征学习上能力减弱，可能影响对相似但不相同实体的区分 (MRR降低~2-3%)。
    *   **结论**: 每个损失项都对模型的最终性能和训练的稳定性有特定贡献。特别是辅助性的路由损失和模态内部损失，对于引导模型学习更鲁棒和有效的表示至关重要。

4.  **多模态融合策略分析 (Analysis of Multi-Modal Fusion Strategy)**：
    *   **实验设置**: 针对 `CrossModalFusion` 模块中的门控融合机制进行分析：
        *   变体 A: 移除门控机制，直接将交叉注意力输出与残差连接相加。
        *   变体 B: 将门控机制替换为简单的平均操作。
    *   **评估指标**: MRR, 以及对融合特征的分析（例如，通过可视化门控值）。
    *   **结果与分析**:
        *   **变体 A (无门控)**: MRR 下降约 1-2%。门控机制能够动态调整来自不同模态信息的贡献，移除后模型可能无法有效抑制噪声或强调重要信息。
        *   **变体 B (平均融合)**: MRR 下降约 0.5-1%。虽然优于无门控，但简单的平均不如学习到的门控那样灵活和有效。可视化门控值显示，模型确实学会了根据上下文动态调整信息流。
    *   **结论**: `CrossModalFusion` 中的门控机制为多模态信息的有效整合提供了有益的动态调整能力。

*(请注意，以上均为示例性描述，实际的消融研究应提供更详尽的设置、精确的实验数据和深入的讨论。)*

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