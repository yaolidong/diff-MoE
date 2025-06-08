import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from PIL import Image  # Ensure PIL.Image is imported
from transformers import CLIPTokenizer

from config import KGAlignmentDatasetConfig
from data_utils import load_alignment_pairs, load_entity_text_attributes, load_entity_image_paths

# 在文件顶部添加缓存字典
_DESCRIPTIONS_CACHE = {}



# CIFAR-10数据集的类别描述
CIFAR10_DESCRIPTIONS = {
    0: "一架飞机在天空中飞行，这是一种用于空中运输的机械飞行器。",
    1: "一辆汽车在道路上行驶，这是一种四轮机动车辆，用于陆地运输。",
    2: "一只鸟站在树枝上，这是一种有羽毛、翅膀和喙的温血动物。",
    3: "一只猫坐在地板上，这是一种常见的家养宠物，有柔软的毛发和尖尖的耳朵。",
    4: "一只鹿站在草地上，这是一种有蹄类哺乳动物，通常有分叉的角。",
    5: "一只狗站在草地上，这是一种忠诚的家养宠物，是人类最好的朋友。",
    6: "一只青蛙在水边，这是一种两栖动物，有光滑的皮肤和长腿。",
    7: "一匹马在草原上奔跑，这是一种大型哺乳动物，常被用于骑乘和拉车。",
    8: "一艘船在水面上航行，这是一种用于水上运输的船只。",
    9: "一辆卡车在公路上行驶，这是一种用于运输货物的大型车辆。"
}

# Fashion-MNIST数据集的类别描述
FASHION_MNIST_DESCRIPTIONS = {
    0: "一件T恤衫，这是一种简单的上衣，通常由棉质面料制成，适合日常穿着。",
    1: "一条裤子，这是一种下装，覆盖腿部的服装，有各种款式和长度。",
    2: "一件套头衫，这是一种保暖的上衣，通常有长袖和高领。",
    3: "一件连衣裙，这是一种女性服装，上下连为一体，长度通常及膝或以下。",
    4: "一件外套，这是一种穿在其他衣服外面的服装，用于保暖或防雨。",
    5: "一只凉鞋，这是一种夏季鞋类，通常露出脚趾和脚跟。",
    6: "一件衬衫，这是一种正式或休闲的上衣，通常有领子和纽扣。",
    7: "一只运动鞋，这是一种适合运动和日常穿着的舒适鞋类。",
    8: "一个手提包，这是一种用于携带个人物品的袋子，有手柄或肩带。",
    9: "一只踝靴，这是一种覆盖脚踝的短靴，适合各种场合穿着。"
}

# Flickr8k数据集的类别描述
FLICKR8K_DESCRIPTIONS = {
    0: "图中有人物，可能是成人或儿童，在进行各种活动或摆姿势。",
    1: "图中有动物，可能是宠物或野生动物，如狗、猫、鸟或其他生物。",
    2: "图中展示体育或运动场景，人们可能在进行跑步、游泳、球类运动等活动。",
    3: "图中是自然风景，可能包括山脉、树木、花朵、草地等自然元素。",
    4: "图中是城市场景，可能包括建筑物、街道、广场等都市元素。",
    5: "图中是室内场景，展示了房间、家具、室内装饰等物品。",
    6: "图中有交通工具，如汽车、自行车、飞机、船只等运输设备。",
    7: "图中有水域场景，如海洋、湖泊、河流、瀑布等水体。"
}

def get_text_descriptions(dataset_name):
    """获取数据集的文本描述"""
    if dataset_name.lower() in _DESCRIPTIONS_CACHE:
        return _DESCRIPTIONS_CACHE[dataset_name.lower()]
    
    if dataset_name.lower() == 'cifar10':
        _DESCRIPTIONS_CACHE[dataset_name.lower()] = CIFAR10_DESCRIPTIONS
        return CIFAR10_DESCRIPTIONS
    elif dataset_name.lower() in ['fashion-mnist', 'fashion_mnist']:
        _DESCRIPTIONS_CACHE['fashion_mnist'] = FASHION_MNIST_DESCRIPTIONS
        return FASHION_MNIST_DESCRIPTIONS
    elif dataset_name.lower() == 'flickr8k':
        _DESCRIPTIONS_CACHE[dataset_name.lower()] = FLICKR8K_DESCRIPTIONS
        return FLICKR8K_DESCRIPTIONS
    else:
        _DESCRIPTIONS_CACHE[dataset_name.lower()] = {}
        return {}

#------------------------------------------------------------------------
# 数据集类定义部分
#------------------------------------------------------------------------

class TextEnhancedDataset(Dataset):
    """文本增强数据集
    
    为单模态数据集添加文本描述，以支持跨模态训练
    """
    def __init__(self, dataset, text_descriptions, max_length=77):
        """
        初始化文本增强数据集
        
        Args:
            dataset: 原始数据集
            text_descriptions: 类别到文本描述的映射字典
            max_length: 文本序列的最大长度
        """
        self.dataset = dataset
        self.text_descriptions = text_descriptions
        self.max_length = max_length
        self.tokenizer = self._get_tokenizer()
        print(f"创建文本增强数据集，包含{len(dataset)}个样本")
        
    def _get_tokenizer(self):
        """获取CLIP tokenizer"""
        return Flickr8kDataset.get_clip_tokenizer()
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.dataset)
        
    def __getitem__(self, idx):
        """获取指定索引的样本，并添加文本描述
        
        Args:
            idx: 样本索引
            
        Returns:
            图像张量、文本token、注意力掩码和标签的元组
        """
        # 获取原始样本
        sample = self.dataset[idx]
        
        if isinstance(sample, tuple) and len(sample) >= 2:
            # 处理常规数据集，返回(image, label)
            image = sample[0]
            label = sample[1]
            
            # 获取对应标签的文本描述
            text = self.text_descriptions.get(int(label), "无描述")
            
            # 对文本进行编码
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 去掉批次维度
            input_ids = tokens.input_ids.squeeze(0)
            attention_mask = tokens.attention_mask.squeeze(0)
            
            return image, input_ids, attention_mask, label
        else:
            # 处理其他格式的数据集
            return sample

class Flickr8kDataset(Dataset):
    """Flickr8k数据集类
    
    加载Flickr8k数据集图像和文本描述
    """
    
    def __init__(self, root: str, split: str = 'train', transform = None):
        """
        初始化Flickr8k数据集
        
        Args:
            root: 数据集根目录
            split: 数据集分割，'train'或'test'
            transform: 图像变换
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # 创建目录结构
        os.makedirs(os.path.join(root, 'images'), exist_ok=True)
        
        # 加载数据
        self.data = self._load_dataset()
        
        # 设置类别标签映射
        self.categories = list(FLICKR8K_DESCRIPTIONS.keys())
        self.class_names = list(FLICKR8K_DESCRIPTIONS.values())
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """
        加载数据集
        
        Returns:
            数据列表，每个元素是包含图像路径、标题和类别的字典
        """
        metadata_path = os.path.join(self.root, f'flickr8k_{self.split}_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        获取指定索引的样本
        
        Args:
            idx: 样本索引
            
        Returns:
            图像张量、标题字符串和类别索引的元组
        """
        # 获取样本数据
        sample = self.data[idx]
        image_path = sample['image_path']
        captions = sample['captions']
        category = sample['category']
        
        # 选择一个标题
        caption = captions[0] if captions else "无描述"
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            image = transform(image)
        
        # 返回图像张量、标题字符串和类别索引
        return image, caption, category
        
    def get_categories(self) -> List[str]:
        """获取所有类别标签"""
        return self.class_names
        
    @staticmethod
    def get_clip_tokenizer(cache_dir='./clip_cache'):
        """获取CLIP tokenizer实例
        
        Args:
            cache_dir: 缓存目录
            
        Returns:
            CLIPTokenizer实例
        """
        try:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", 
                                                    local_files_only=True,
                                                    cache_dir=cache_dir)
        except:
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",
                                                    cache_dir=cache_dir)
        return tokenizer
        
    def tokenize_captions(self, captions, max_length=77):
        """将标题文本转换为token
        
        Args:
            captions: 字符串或字符串列表
            max_length: 最大序列长度
            
        Returns:
            包含input_ids和attention_mask的字典
        """
        tokenizer = self.get_clip_tokenizer()
        return tokenizer(
            captions, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )

class KGAlignmentDataset(Dataset):
    """
    知识图谱对齐数据集 (Knowledge Graph Alignment Dataset)

    加载实体对、它们的文本属性和图像。
    """
    def __init__(self,
                 config: KGAlignmentDatasetConfig,
                 split: str,
                 tokenizer, # 外部传入的tokenizer, e.g., from transformers
                 transform, # 外部传入的图像变换
                 text_tokenizer_max_len: int = 32):
        """
        初始化KGAlignmentDataset

        Args:
            config: KGAlignmentDatasetConfig 实例
            split: 数据集分割 ('train', 'val', 'test')
            tokenizer: 用于文本编码的tokenizer
            transform: 用于图像预处理的 torchvision transform
            text_tokenizer_max_len: tokenizer处理文本时的最大长度
        """
        self.config = config
        self.split = split.lower()
        self.tokenizer = tokenizer
        self.transform = transform
        self.text_tokenizer_max_len = text_tokenizer_max_len

        # 1. 加载对齐实体对
        self.aligned_pairs: List[Tuple[str, str]] = []
        alignment_file = None
        if self.split == 'train':
            alignment_file = self.config.alignment_train_file
        elif self.split == 'val':
            alignment_file = self.config.alignment_val_file
        elif self.split == 'test':
            alignment_file = self.config.alignment_test_file

        if alignment_file:
            if not os.path.exists(alignment_file):
                print(f"Warning: Alignment file for split '{self.split}' not found at {alignment_file}. Dataset will be empty.")
            else:
                try:
                    self.aligned_pairs = load_alignment_pairs(alignment_file)
                except FileNotFoundError:
                    print(f"Error: Alignment file for split '{self.split}' not found at {alignment_file}. Dataset will be empty.")
        else:
            print(f"Warning: No alignment file specified for split '{self.split}' in config. Dataset will be empty.")

        # 2. 加载实体文本属性
        self.entity_texts: Dict[str, str] = {}
        if self.config.entity_text_file:
            if not os.path.exists(self.config.entity_text_file):
                print(f"Warning: Entity text file not found at {self.config.entity_text_file}. Text attributes will be empty.")
            else:
                try:
                    self.entity_texts = load_entity_text_attributes(self.config.entity_text_file)
                except FileNotFoundError:
                     print(f"Error: Entity text file not found at {self.config.entity_text_file}. Text attributes will be empty.")
        else:
            print("Info: No entity text file specified in config. Text attributes will be empty.")

        # 3. 收集所有独特的实体ID，并加载图像路径
        all_entity_ids_set = set()
        for e1, e2 in self.aligned_pairs:
            all_entity_ids_set.add(e1)
            all_entity_ids_set.add(e2)
        all_entity_ids_list = list(all_entity_ids_set)

        self.entity_image_paths: Dict[str, str] = {}
        if self.config.entity_img_dir:
            if not os.path.isdir(self.config.entity_img_dir):
                print(f"Warning: Entity image directory not found at {self.config.entity_img_dir}. Image paths will be empty.")
            else:
                try:
                    self.entity_image_paths = load_entity_image_paths(self.config.entity_img_dir, all_entity_ids_list)
                except FileNotFoundError:
                    print(f"Error: Entity image directory not found at {self.config.entity_img_dir}. Image paths will be empty.")
        else:
            print("Info: No entity image directory specified in config. Image paths will be empty.")

        # 4. 初始化默认图像张量 (使用config中的in_channels和image_size)
        # DatasetConfig (父类) 应该有 in_channels 和 image_size
        if not hasattr(self.config, 'in_channels') or not hasattr(self.config, 'image_size'):
            raise ValueError("KGAlignmentDatasetConfig must have 'in_channels' and 'image_size' attributes (possibly inherited).")

        # image_size is expected to be a tuple (height, width)
        if not isinstance(self.config.image_size, tuple) or len(self.config.image_size) != 2:
             raise ValueError(f"config.image_size must be a tuple of (height, width), got {self.config.image_size}")

        self.default_image_tensor = torch.zeros(
            self.config.in_channels,
            self.config.image_size[0],
            self.config.image_size[1]
        )

        print(f"KGAlignmentDataset for split '{self.split}' initialized: {len(self.aligned_pairs)} pairs.")
        if not self.aligned_pairs:
             print(f"Warning: KGAlignmentDataset for split '{self.split}' is empty. Check file paths and content.")


    def __len__(self) -> int:
        """返回对齐实体对的数量"""
        return len(self.aligned_pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
        """
        获取指定索引的样本对。

        返回:
            (img_tensor1, input_ids1, attention_mask1,
             img_tensor2, input_ids2, attention_mask2,
             label)
        """
        if idx >= len(self.aligned_pairs):
            raise IndexError("Index out of bounds")

        entity1_id, entity2_id = self.aligned_pairs[idx]

        # 处理实体1
        text1 = self.entity_texts.get(entity1_id, "")
        tokenized_text1 = self.tokenizer(
            text1,
            padding='max_length',
            truncation=True,
            max_length=self.text_tokenizer_max_len,
            return_tensors='pt'
        )
        input_ids1 = tokenized_text1['input_ids'].squeeze(0) # Remove batch dim
        attention_mask1 = tokenized_text1['attention_mask'].squeeze(0) # Remove batch dim

        img_path1 = self.entity_image_paths.get(entity1_id)
        if img_path1 and os.path.exists(img_path1):
            try:
                image1 = Image.open(img_path1).convert('RGB')
                img_tensor1 = self.transform(image1) if self.transform else self.default_image_tensor
            except Exception as e:
                print(f"Warning: Error loading image {img_path1} for entity {entity1_id}: {e}. Using default tensor.")
                img_tensor1 = self.default_image_tensor
        else:
            img_tensor1 = self.default_image_tensor

        # 处理实体2
        text2 = self.entity_texts.get(entity2_id, "")
        tokenized_text2 = self.tokenizer(
            text2,
            padding='max_length',
            truncation=True,
            max_length=self.text_tokenizer_max_len,
            return_tensors='pt'
        )
        input_ids2 = tokenized_text2['input_ids'].squeeze(0) # Remove batch dim
        attention_mask2 = tokenized_text2['attention_mask'].squeeze(0) # Remove batch dim

        img_path2 = self.entity_image_paths.get(entity2_id)
        if img_path2 and os.path.exists(img_path2):
            try:
                image2 = Image.open(img_path2).convert('RGB')
                img_tensor2 = self.transform(image2) if self.transform else self.default_image_tensor
            except Exception as e:
                print(f"Warning: Error loading image {img_path2} for entity {entity2_id}: {e}. Using default tensor.")
                img_tensor2 = self.default_image_tensor
        else:
            img_tensor2 = self.default_image_tensor

        # 标签 (默认为正样本对)
        # TODO: Implement negative sampling strategy if needed, e.g., by modifying label or pair
        label = torch.tensor(1, dtype=torch.long)

        return (img_tensor1, input_ids1, attention_mask1,
                img_tensor2, input_ids2, attention_mask2,
                label)
