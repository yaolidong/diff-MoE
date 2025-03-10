import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from transformers import CLIPTokenizer

# 在文件顶部添加缓存字典
_DESCRIPTIONS_CACHE = {}

#------------------------------------------------------------------------
# 标签到文本描述的转换部分 (来自原 label_to_text.py)
#------------------------------------------------------------------------

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
    """
    根据数据集名称获取类别的文本描述，使用缓存避免重复加载
    
    Args:
        dataset_name: 数据集名称，如'cifar10'或'fashion-mnist'
        
    Returns:
        包含类别描述的字典，键为类别索引，值为文本描述
    """
    # 如果缓存中已有该数据集的描述，直接返回
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

def get_enhanced_descriptions(dataset_name, detail_level='normal'):
    """
    获取增强的文本描述，可以根据需要的详细程度调整
    
    Args:
        dataset_name: 数据集名称
        detail_level: 详细程度，可以是'simple'、'normal'或'detailed'
        
    Returns:
        增强的文本描述字典
    """
    base_descriptions = get_text_descriptions(dataset_name)
    
    if detail_level == 'simple':
        # 返回简化版描述
        return {k: v.split('，')[0] for k, v in base_descriptions.items()}
    
    elif detail_level == 'detailed':
        # 这里可以返回更详细的描述
        # 在实际应用中，可以从更大的文本语料库中获取
        detailed_descriptions = {}
        for k, v in base_descriptions.items():
            detailed_descriptions[k] = v + " 这是一个常见的物体，在日常生活中经常可以看到。它有独特的特征和用途。"
        return detailed_descriptions
    
    else:  # 'normal'
        return base_descriptions

def tokenize_text(text, max_length=32):
    """
    简单的文本标记化函数（示例）
    
    在实际应用中，应该使用专业的分词器如BERT、GPT等的tokenizer
    
    Args:
        text: 要标记化的文本
        max_length: 最大标记长度
        
    Returns:
        标记化后的文本（这里简化为字符索引）
    """
    # 这只是一个非常简化的示例
    # 实际应用中应该使用专业的tokenizer
    chars = list(text[:max_length])
    # 将字符转换为简单的数字索引（仅用于演示）
    tokens = [ord(c) % 1000 for c in chars]
    # 填充到最大长度
    tokens = tokens + [0] * (max_length - len(tokens))
    return tokens


#------------------------------------------------------------------------
# 数据集类定义部分
#------------------------------------------------------------------------

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
        # 加载数据
        try:
            # 尝试加载真实数据
            metadata_path = os.path.join(self.root, f'flickr8k_{self.split}_metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return metadata
            
        except FileNotFoundError:
            # 创建模拟数据以进行测试
            mock_data = []
            num_samples = 100 if self.split == 'train' else 20
            
            for i in range(num_samples):
                # 随机选择一个类别
                category_idx = np.random.randint(0, len(FLICKR8K_DESCRIPTIONS))
                
                # 创建一个模拟的图像路径
                image_id = f"mock_image_{i:05d}.jpg"
                image_path = os.path.join(self.root, 'images', image_id)
                
                # 创建模拟的标题
                category = self.class_names[category_idx]
                captions = [f"这是一张关于{category}的图片，场景包含了多个元素。"]
                
                # 添加到数据列表
                mock_data.append({
                    'image_path': image_path,
                    'captions': captions,
                    'category': category_idx
                })
            
            # 保存模拟数据
            os.makedirs(os.path.dirname(os.path.join(self.root, f'flickr8k_{self.split}_metadata.json')), exist_ok=True)
            with open(os.path.join(self.root, f'flickr8k_{self.split}_metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(mock_data, f, ensure_ascii=False, indent=2)
            
            return mock_data
    
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
        
        try:
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
                
        except FileNotFoundError:
            # 创建随机图像
            random_image = torch.randn(3, 224, 224)
            # 应用默认归一化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            random_image = random_image * std + mean
            image = random_image
            
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