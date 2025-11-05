from typing import List, Dict, Tuple, Sequence, Optional, Union
import os

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .dataset_mimic import MimicDataset
from .dataset_chexpert import CheXpertDataset
from .dataset_padchest import PadChestDataset


class CombinedCollator():
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


def _build_single_dataset(
    data_path: str,
    data_format: str,
    img_size: int = 224,
    is_train: bool = True,
    text_column: str = None,
    image_column: str = None,
    image_dir: str = None,
    split: str = "train",
    dataset_name: str = None,
) -> Dataset:
    """根据配置构建单个子数据集。

    Args:
        data_path: 数据文件路径
        data_format: 数据格式 ("json", "csv", "txt", "folder")
        img_size: 图像尺寸
        is_train: 是否为训练模式
        text_column: 文本列名
        image_column: 图像路径列名
        image_dir: 图像根目录路径
        split: 数据集分割（train, validate）
        dataset_name: 数据集名称（必须提供，不能为None）
    
    Returns:
        构建的数据集实例
    """
    if not data_path:
        raise ValueError("data_path 不能为空")
    
    if dataset_name is None:
        raise ValueError("dataset_name 必须提供，不能为 None")
    
    dataset_name = dataset_name.lower()
    
    # 根据数据集类型设置默认值
    if dataset_name == 'mimic':
        default_text_column = text_column or 'report'
        default_image_column = image_column or 'image_path'
        default_format = data_format or 'json'
    elif dataset_name == 'chexpert':
        default_text_column = text_column or 'findings_section'
        default_image_column = image_column or 'image_paths'
        default_format = data_format or 'csv'
    elif dataset_name == 'padchest':
        default_text_column = text_column or 'report'
        default_image_column = image_column or 'image_path'
        default_format = data_format or 'csv'
    else:
        raise ValueError(f"未知的数据集类型: {dataset_name}. 期望 'mimic', 'chexpert' 或 'padchest'.")
    
    common_kwargs = {
        'data_path': data_path,
        'data_format': default_format,
        'image_size': img_size,
        'is_train': is_train,
        'text_column': default_text_column,
        'image_column': default_image_column,
        'image_dir': image_dir,
        'split': split,
    }
    
    if dataset_name == 'mimic':
        return MimicDataset(**common_kwargs)
    elif dataset_name == 'chexpert':
        return CheXpertDataset(**common_kwargs)
    elif dataset_name == 'padchest':
        return PadChestDataset(**common_kwargs)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_name}")


def get_combined_dataset_and_collator(
    data_path: Union[str, List[str]],
    data_format: Union[str, List[str]],
    img_size: int = 224,
    is_train: bool = True,
    text_column: Union[str, List[str]] = None,
    image_column: Union[str, List[str]] = None,
    image_dir: Union[str, List[str]] = None,
    split: str = "train",
    dataset_name: Union[str, List[str]] = None,
) -> Tuple[Dataset, CombinedCollator]:
    """根据列表参数构建多个数据集并合并。

    Args:
        data_path: 数据文件路径（字符串或列表）
        data_format: 数据格式（字符串或列表）
        img_size: 图像尺寸
        is_train: 是否为训练模式
        text_column: 文本列名（字符串或列表）
        image_column: 图像路径列名（字符串或列表）
        image_dir: 图像根目录路径（字符串或列表）
        split: 数据集分割
        dataset_name: 数据集名称（字符串或列表，用于指定每个数据集类型）
    
    Returns:
        (合并后的数据集, 数据整理器)
    """
    # 将单个值转换为列表
    if isinstance(data_path, str):
        data_path = [data_path]
    if isinstance(data_format, str):
        data_format = [data_format]
    if text_column is None:
        text_column = [None] * len(data_path)
    elif isinstance(text_column, str):
        text_column = [text_column]
    if image_column is None:
        image_column = [None] * len(data_path)
    elif isinstance(image_column, str):
        image_column = [image_column]
    if image_dir is None:
        image_dir = [None] * len(data_path)
    elif isinstance(image_dir, str):
        image_dir = [image_dir]
    if dataset_name is None:
        raise ValueError("dataset_name 必须提供，不能为 None")
    elif isinstance(dataset_name, str):
        dataset_name = [dataset_name]
    
    # 确保所有列表长度一致
    n_datasets = len(data_path)
    if len(data_format) != n_datasets:
        raise ValueError(f"data_format 列表长度 ({len(data_format)}) 与 data_path 列表长度 ({n_datasets}) 不匹配")
    if len(text_column) != n_datasets:
        text_column = text_column * (n_datasets // len(text_column)) if len(text_column) > 0 else [None] * n_datasets
    if len(image_column) != n_datasets:
        image_column = image_column * (n_datasets // len(image_column)) if len(image_column) > 0 else [None] * n_datasets
    if len(image_dir) != n_datasets:
        image_dir = image_dir * (n_datasets // len(image_dir)) if len(image_dir) > 0 else [None] * n_datasets
    if len(dataset_name) != n_datasets:
        raise ValueError(f"dataset_name 列表长度 ({len(dataset_name)}) 与 data_path 列表长度 ({n_datasets}) 不匹配")
    
    # 构建每个数据集
    datasets: List[Dataset] = []
    for i in range(n_datasets):
        ds = _build_single_dataset(
            data_path=data_path[i],
            data_format=data_format[i],
            img_size=img_size,
            is_train=is_train,
            text_column=text_column[i],
            image_column=image_column[i],
            image_dir=image_dir[i],
            split=split,
            dataset_name=dataset_name[i],  # 直接使用 dataset_name 中的类型
        )
        datasets.append(ds)
        print(f"Built dataset {i+1}/{n_datasets}: {type(ds).__name__} with {len(ds)} samples")
    
    # 合并数据集
    combined = ConcatDataset(datasets)
    collator = CombinedCollator()
    print(f"Combined dataset total size: {len(combined)}")
    return combined, collator


def loader(
    train_batch_size: int,
    num_workers: int,
    data_path: Union[str, List[str]],
    data_format: Union[str, List[str]] = None,
    img_size: int = 224,
    text_column: Union[str, List[str]] = None,
    image_column: Union[str, List[str]] = None,
    image_dir: Union[str, List[str]] = None,
    split: str = "train",
    dataset_name: Union[str, List[str]] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
):
    """数据加载器函数，支持列表参数以合并多个数据集。

    Args:
        train_batch_size: 训练批次大小
        num_workers: 数据加载器工作进程数
        data_path: 数据文件路径（字符串或列表）
        data_format: 数据格式（字符串或列表）
        img_size: 图像尺寸
        text_column: 文本列名（字符串或列表）
        image_column: 图像路径列名（字符串或列表）
        image_dir: 图像根目录路径（字符串或列表）
        split: 数据集分割
        dataset_name: 数据集名称（字符串或列表，用于指定每个数据集类型）
        shuffle: 是否打乱数据
        drop_last: 是否丢弃最后不完整的批次
        **kwargs: 其他参数（会被忽略）
    
    Returns:
        DataLoader实例
    """
    dataset, collator = get_combined_dataset_and_collator(
        data_path=data_path,
        data_format=data_format,
        img_size=img_size,
        is_train=True,
        text_column=text_column,
        image_column=image_column,
        image_dir=image_dir,
        split=split,
        dataset_name=dataset_name,
    )
    
    return DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        collate_fn=collator,
        shuffle=shuffle,
        drop_last=drop_last,
    )


