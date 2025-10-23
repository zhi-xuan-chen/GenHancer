import os
import json
import random
from typing import Optional, Sequence, Dict, Union, Tuple, List
from dataclasses import dataclass
from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
import glob


OPENAI_DATASET_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_DATASET_STD = [0.26862954, 0.26130258, 0.27577711]

DEFAULT_IMAGE_FILE_SUFFIX = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']


def _convert_to_rgb(image):
    """Convert image to RGB format."""
    try:
        image = image.convert('RGB')
        return image
    except Exception as e:
        print(f"Error converting image to RGB: {e}")
        return image


def to_tensor(image):
    """Convert PIL image to tensor and normalize to [0, 1]."""
    return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0


def image_transform(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
):
    """Create image transformation pipeline."""
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        return Compose([
            RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
            _convert_to_rgb,
            to_tensor,
        ])
    else:
        return Compose([
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _convert_to_rgb,
            to_tensor,
        ])


class CustomDataset(Dataset):
    """自定义数据集类，支持多种数据格式"""
    
    def __init__(self, data_path: str, data_format: str = "json", 
                 image_size: int = 224, is_train: bool = True, 
                 text_column: str = "caption", image_column: str = "image_path",
                 image_dir: str = None):
        """
        Args:
            data_path: 数据文件路径或目录路径
            data_format: 数据格式 ("json", "csv", "txt", "folder")
            image_size: 图像尺寸
            is_train: 是否为训练模式
            text_column: 文本列名（用于json/csv格式）
            image_column: 图像路径列名（用于json/csv格式）
            image_dir: 图像根目录路径，用于将相对路径转换为绝对路径
        """
        self.data_path = data_path
        self.data_format = data_format
        self.text_column = text_column
        self.image_column = image_column
        self.image_dir = image_dir
        self.transform = image_transform(image_size, is_train)
        
        # 加载数据
        self.data_pairs = self._load_data()
    
    def _get_absolute_image_path(self, image_path: str) -> str:
        """将相对路径转换为绝对路径"""
        if self.image_dir and not os.path.isabs(image_path):
            return os.path.join(self.image_dir, image_path)
        return image_path
        
    def _load_data(self):
        """根据数据格式加载数据"""
        data_pairs = []
        
        if self.data_format == "json":
            data_pairs = self._load_json()
        elif self.data_format == "csv":
            data_pairs = self._load_csv()
        elif self.data_format == "txt":
            data_pairs = self._load_txt()
        elif self.data_format == "folder":
            data_pairs = self._load_folder()
        else:
            raise ValueError(f"不支持的数据格式: {self.data_format}")
            
        return data_pairs
    
    def _load_json(self):
        """加载JSON格式数据"""
        data_pairs = []
        
        if os.path.isfile(self.data_path):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 如果是目录，加载所有json文件
            json_files = glob.glob(os.path.join(self.data_path, "*.json"))
            data = []
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data.extend(json.load(f))
        
        for item in data:
            if self.image_column in item and self.text_column in item:
                image_path = self._get_absolute_image_path(item[self.image_column])
                text = item[self.text_column]
                if os.path.exists(image_path):
                    data_pairs.append((image_path, text))
        
        return data_pairs
    
    def _load_csv(self):
        """加载CSV格式数据"""
        data_pairs = []
        
        if os.path.isfile(self.data_path):
            df = pd.read_csv(self.data_path)
        else:
            # 如果是目录，加载所有csv文件
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            dfs = []
            for csv_file in csv_files:
                dfs.append(pd.read_csv(csv_file))
            df = pd.concat(dfs, ignore_index=True)
        
        for _, row in df.iterrows():
            if self.image_column in row and self.text_column in row:
                image_path = self._get_absolute_image_path(row[self.image_column])
                text = row[self.text_column]
                if os.path.exists(image_path):
                    data_pairs.append((image_path, text))
        
        return data_pairs
    
    def _load_txt(self):
        """加载TXT格式数据（每行格式：image_path|caption）"""
        data_pairs = []
        
        if os.path.isfile(self.data_path):
            txt_files = [self.data_path]
        else:
            txt_files = glob.glob(os.path.join(self.data_path, "*.txt"))
        
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if '|' in line:
                        image_path, text = line.split('|', 1)
                        image_path = self._get_absolute_image_path(image_path.strip())
                        if os.path.exists(image_path):
                            data_pairs.append((image_path, text.strip()))
        
        return data_pairs
    
    def _load_folder(self):
        """加载文件夹格式数据（每个子文件夹包含图像和对应的文本文件）"""
        data_pairs = []
        
        # 遍历所有子文件夹
        for root, dirs, files in os.walk(self.data_path):
            # 查找图像文件
            image_files = []
            for file in files:
                if any(file.lower().endswith(ext) for ext in DEFAULT_IMAGE_FILE_SUFFIX):
                    image_files.append(os.path.join(root, file))
            
            # 为每个图像文件查找对应的文本文件
            for image_path in image_files:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                text_file = os.path.join(root, f"{base_name}.txt")
                
                if os.path.exists(text_file):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    data_pairs.append((image_path, text))
                else:
                    # 如果没有对应的文本文件，使用文件名作为文本
                    text = base_name.replace('_', ' ').replace('-', ' ')
                    data_pairs.append((image_path, text))
        
        return data_pairs
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        image_path, text = self.data_pairs[idx]
        
        try:
            # 加载图像
            image = Image.open(image_path)
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个空白图像作为fallback
            image = torch.zeros(3, 224, 224)
        
        return {
            'image': image,
            'text': text
        }


@dataclass
class CustomDatasetCollator:
    """自定义数据集的数据整理器"""
    
    def __call__(self, samples: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """整理批次数据"""
        images = [sample["image"] for sample in samples]
        texts = [sample["text"] for sample in samples]
        
        batch = {}
        
        # 检查所有图像是否具有相同的形状
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['image'] = torch.stack(images)
        else:
            batch['image'] = images
        
        batch['text'] = texts
        
        return batch


def get_custom_dataset_and_collator(data_path: str, data_format: str = "json",
                                  img_size: int = 224, is_train: bool = True,
                                  text_column: str = "caption", 
                                  image_column: str = "image_path",
                                  image_dir: str = None):
    """获取自定义数据集和数据整理器"""
    
    dataset = CustomDataset(
        data_path=data_path,
        data_format=data_format,
        image_size=img_size,
        is_train=is_train,
        text_column=text_column,
        image_column=image_column,
        image_dir=image_dir
    )
    
    collator = CustomDatasetCollator()
    
    return dataset, collator


def loader(train_batch_size: int, num_workers: int, data_path: str, 
          data_format: str = "json", img_size: int = 224, 
          text_column: str = "caption", image_column: str = "image_path", 
          image_dir: str = None, **args):
    """数据加载器函数，与原始loader接口兼容"""
    
    dataset, collator = get_custom_dataset_and_collator(
        data_path=data_path,
        data_format=data_format,
        img_size=img_size,
        is_train=True,
        text_column=text_column,
        image_column=image_column,
        image_dir=image_dir
    )
    
    return DataLoader(
        dataset, 
        batch_size=train_batch_size, 
        num_workers=num_workers, 
        collate_fn=collator,
        shuffle=True,
        drop_last=True
    )

