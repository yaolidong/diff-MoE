"""
标签到文本描述的转换模块

此模块提供了将数字标签转换为丰富文本描述的功能，
用于增强多模态模型的文本输入。
"""

import logging

logger = logging.getLogger(__name__)

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

def get_text_descriptions(dataset_name):
    """
    根据数据集名称获取类别的文本描述
    
    Args:
        dataset_name: 数据集名称，如'cifar10'或'fashion-mnist'
        
    Returns:
        包含类别描述的字典，键为类别索引，值为文本描述
    """
    if dataset_name.lower() == 'cifar10':
        logger.info("加载CIFAR-10数据集的文本描述")
        return CIFAR10_DESCRIPTIONS
    elif dataset_name.lower() in ['fashion-mnist', 'fashion_mnist']:
        logger.info("加载Fashion-MNIST数据集的文本描述")
        return FASHION_MNIST_DESCRIPTIONS
    else:
        logger.warning(f"未找到数据集'{dataset_name}'的文本描述，返回空字典")
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

if __name__ == "__main__":
    # 简单测试
    logging.basicConfig(level=logging.INFO)
    
    print("CIFAR-10 描述示例:")
    cifar_desc = get_text_descriptions('cifar10')
    for idx, desc in list(cifar_desc.items())[:3]:  # 只打印前3个
        print(f"类别 {idx}: {desc}")
    
    print("\nFashion-MNIST 描述示例:")
    fashion_desc = get_text_descriptions('fashion-mnist')
    for idx, desc in list(fashion_desc.items())[:3]:  # 只打印前3个
        print(f"类别 {idx}: {desc}")
    
    print("\n不同详细程度的描述示例:")
    simple_desc = get_enhanced_descriptions('cifar10', 'simple')
    normal_desc = get_enhanced_descriptions('cifar10', 'normal')
    detailed_desc = get_enhanced_descriptions('cifar10', 'detailed')
    
    print(f"简单描述: {simple_desc[0]}")
    print(f"普通描述: {normal_desc[0]}")
    print(f"详细描述: {detailed_desc[0]}") 