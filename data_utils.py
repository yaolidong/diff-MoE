import os
import requests
import zipfile
import shutil
import argparse
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Flickr8k数据集下载链接
FLICKR8K_IMAGES_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
FLICKR8K_TEXT_URL = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

def download_file(url: str, output_path: str) -> bool:
    """
    下载文件到指定路径
    
    Args:
        url: 下载链接
        output_path: 保存路径
        
    Returns:
        bool: 是否下载成功
    """
    try:
        # 创建目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 检查文件是否已经存在
        if os.path.exists(output_path):
            return True
        
        # 下载文件
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        # 使用tqdm显示进度条
        with open(output_path, 'wb') as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                f.write(data)
                pbar.update(len(data))
        
        return True
    
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    解压ZIP文件
    
    Args:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录
        
    Returns:
        bool: 是否解压成功
    """
    try:
        # 创建解压目录
        os.makedirs(extract_to, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取文件总数
            total_files = len(zip_ref.namelist())
            
            # 使用tqdm显示进度条
            for file in tqdm(zip_ref.namelist(), desc=f"解压 {os.path.basename(zip_path)}", total=total_files):
                zip_ref.extract(file, extract_to)
        
        return True
    
    except Exception as e:
        return False

def organize_dataset(dataset_dir: str, output_dir: str) -> bool:
    """
    整理Flickr8k数据集文件
    
    Args:
        dataset_dir: 解压后的数据集目录
        output_dir: 输出目录
        
    Returns:
        bool: 是否整理成功
    """
    try:
        # 创建输出目录
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # 寻找图像文件夹
        possible_image_dirs = [
            os.path.join(dataset_dir, "Flickr8k_Dataset"),
            os.path.join(dataset_dir, "Flicker8k_Dataset"),
            os.path.join(dataset_dir, "flickr8k", "images"),
            os.path.join(dataset_dir, "flickr8k", "Flicker8k_Dataset"),
            os.path.join(dataset_dir, "Flickr8k_Dataset", "Flicker8k_Dataset"),
        ]
        
        image_dir = None
        for dir_path in possible_image_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                image_dir = dir_path
                break
        
        if not image_dir:
            return False
        
        # 复制图像文件
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in tqdm(image_files, desc="复制图像"):
            src_path = os.path.join(image_dir, image_file)
            dst_path = os.path.join(images_dir, image_file)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
        
        # 寻找文本文件夹
        possible_text_dirs = [
            os.path.join(dataset_dir, "Flickr8k_text"),
            os.path.join(dataset_dir, "flickr8k", "text"),
            os.path.join(dataset_dir, "Flickr8k_text", "Flickr8k_text"),
        ]
        
        text_dir = None
        for dir_path in possible_text_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                text_dir = dir_path
                break
        
        if not text_dir:
            return False
        
        # 读取Flickr_8k.lemma.token.txt文件
        token_file = os.path.join(text_dir, "Flickr8k.lemma.token.txt")
        if not os.path.exists(token_file):
            token_file = os.path.join(text_dir, "Flickr8k.token.txt")
            
        if not os.path.exists(token_file):
            return False
        
        # 解析文件，提取图像标题
        image_captions = {}
        with open(token_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    image_id = parts[0].split('#')[0].strip()
                    caption = parts[1].strip()
                    
                    if image_id not in image_captions:
                        image_captions[image_id] = []
                    
                    image_captions[image_id].append(caption)
        
        # 读取训练/测试分割文件
        train_file = os.path.join(text_dir, "Flickr_8k.trainImages.txt")
        test_file = os.path.join(text_dir, "Flickr_8k.testImages.txt")
        
        train_images = set()
        test_images = set()
        
        # 读取训练集图像ID
        if os.path.exists(train_file):
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    image_id = line.strip()
                    if image_id:
                        train_images.add(image_id)
        
        # 读取测试集图像ID
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    image_id = line.strip()
                    if image_id:
                        test_images.add(image_id)
        
        # 如果没有分割文件，随机分割
        if not train_images and not test_images:
            all_images = list(image_captions.keys())
            random.shuffle(all_images)
            
            # 80%训练，20%测试
            split_idx = int(len(all_images) * 0.8)
            train_images = set(all_images[:split_idx])
            test_images = set(all_images[split_idx:])
        
        # 创建元数据
        train_metadata = []
        test_metadata = []
        
        # 为每个图像分配一个场景类别（这里简化处理，实际使用时应该有更好的分类方法）
        scene_categories = [
            '人物', '动物', '运动', '自然风景', '城市场景',
            '室内场景', '交通工具', '水域场景'
        ]
        
        # 处理训练集
        for image_id in tqdm(train_images, desc="处理训练集"):
            if image_id in image_captions:
                # 随机分配一个类别（仅用于演示）
                category = random.randint(0, len(scene_categories) - 1)
                
                train_metadata.append({
                    'image_path': os.path.join("images", image_id),
                    'captions': image_captions[image_id],
                    'category': category
                })
        
        # 处理测试集
        for image_id in tqdm(test_images, desc="处理测试集"):
            if image_id in image_captions:
                # 随机分配一个类别（仅用于演示）
                category = random.randint(0, len(scene_categories) - 1)
                
                test_metadata.append({
                    'image_path': os.path.join("images", image_id),
                    'captions': image_captions[image_id],
                    'category': category
                })
        
        # 保存元数据
        train_metadata_file = os.path.join(output_dir, "flickr8k_train_metadata.json")
        test_metadata_file = os.path.join(output_dir, "flickr8k_test_metadata.json")
        
        with open(train_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(train_metadata, f, ensure_ascii=False, indent=2)
        
        with open(test_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(test_metadata, f, ensure_ascii=False, indent=2)
        
        return True
    
    except Exception as e:
        return False

def download_flickr8k(download_dir: str, output_dir: str, skip_download: bool = False) -> bool:
    """
    下载并处理Flickr8k数据集
    
    Args:
        download_dir: 下载目录
        output_dir: 输出目录
        skip_download: 是否跳过下载过程
        
    Returns:
        bool: 是否成功
    """
    try:
        # 创建目录
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义文件路径
        images_zip = os.path.join(download_dir, "Flickr8k_Dataset.zip")
        text_zip = os.path.join(download_dir, "Flickr8k_text.zip")
        extract_dir = os.path.join(download_dir, "extracted")
        
        # 下载数据
        if not skip_download:
            # 下载图像
            if not download_file(FLICKR8K_IMAGES_URL, images_zip):
                return False
            
            # 下载文本
            if not download_file(FLICKR8K_TEXT_URL, text_zip):
                return False
        
        # 解压文件
        # 解压图像
        if not extract_zip(images_zip, extract_dir):
            return False
        
        # 解压文本
        if not extract_zip(text_zip, extract_dir):
            return False
        
        # 整理数据集
        if not organize_dataset(extract_dir, output_dir):
            return False
        
        return True
        
    except Exception as e:
        return False

# 其他数据集下载和处理函数
def download_cifar10(download_dir: str) -> str:
    """
    下载CIFAR10数据集
    直接使用torchvision的内置功能
    
    Args:
        download_dir: 下载目录
        
    Returns:
        str: 数据集路径
    """
    try:
        os.makedirs(download_dir, exist_ok=True)
        
        # 使用torchvision下载
        _ = datasets.CIFAR10(
            root=download_dir,
            train=True,
            download=True
        )
        
        return os.path.join(download_dir, 'cifar-10-batches-py')
    
    except Exception as e:
        return ""

def download_fashion_mnist(download_dir: str) -> str:
    """
    下载Fashion-MNIST数据集
    直接使用torchvision的内置功能
    
    Args:
        download_dir: 下载目录
        
    Returns:
        str: 数据集路径
    """
    try:
        os.makedirs(download_dir, exist_ok=True)
        
        # 使用torchvision下载
        _ = datasets.FashionMNIST(
            root=download_dir,
            train=True,
            download=True
        )
        
        return os.path.join(download_dir, 'FashionMNIST')
    
    except Exception as e:
        return ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载并处理数据集')
    parser.add_argument('--dataset', type=str, default='flickr8k', choices=['flickr8k', 'cifar10', 'fashion_mnist'],
                      help='要下载的数据集')
    parser.add_argument('--download_dir', type=str, default='downloads',
                      help='下载目录')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='输出目录')
    parser.add_argument('--skip_download', action='store_true',
                      help='是否跳过下载过程')
    
    args = parser.parse_args()
    
    if args.dataset == 'flickr8k':
        download_flickr8k(args.download_dir, os.path.join(args.output_dir, 'Flickr8k'), args.skip_download)
    elif args.dataset == 'cifar10':
        download_cifar10(os.path.join(args.output_dir, 'CIFAR10'))
    elif args.dataset == 'fashion_mnist':
        download_fashion_mnist(os.path.join(args.output_dir, 'FashionMNIST')) 