import torchvision.transforms as transforms
import random

class ImageAugmentation:
    def __init__(self):
        self.augmentations = [
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # FashionMNIST 的均值和标准差
        ])

    def __call__(self, x):
        for aug in self.augmentations:
            x = aug(x)
        return self.normalize(x)

def get_augmentation():
    return ImageAugmentation()
