def fashion_mnist_label_to_text(label: int) -> str:
    """
    将Fashion-MNIST数据集的标签转换为文本描述
    
    Args:
        label: int, 标签索引
        
    Returns:
        str: 对应的文本描述
    """
    return {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }[label]

def cifar10_label_to_text(label: int) -> str:
    """
    将CIFAR-10数据集的标签转换为文本描述
    
    Args:
        label: int, 标签索引
        
    Returns:
        str: 对应的文本描述
    """
    return {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }[label] 