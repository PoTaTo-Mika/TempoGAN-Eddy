# Dataset for digital typhoon
import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
from typing import List, Tuple, Optional

class DigitalTyphoonDataset(Dataset):
    """
    Digital Typhoon数据集类，用于超分辨率任务
    支持时序图像处理，可以生成连续三帧用于时间判别器
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 scale_factor: int = 4,
                 patch_size: int = 64,
                 sequence_length: int = 3,
                 augment: bool = True):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录
            split: 数据集分割 ('train', 'val', 'test')
            scale_factor: 超分辨率倍数
            patch_size: 训练时的图像块大小
            sequence_length: 时序序列长度（用于时间判别器）
            augment: 是否使用数据增强
        """
        self.data_root = data_root
        self.split = split
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.augment = augment
        
        # 数据路径列表
        self.data_paths = self._load_data_paths()
        
        # 时序分组：将连续的图像分组
        self.sequence_groups = self._group_sequences()
        
        print(f"Loaded {len(self.data_paths)} images for {split} split")
        print(f"Created {len(self.sequence_groups)} sequence groups")
    
    def _load_data_paths(self) -> List[str]:
        """加载数据路径"""
        data_paths = []
        
        # 根据split确定子目录
        split_dir = os.path.join(self.data_root, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # 遍历所有图像文件
        for root, dirs, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    data_paths.append(os.path.join(root, file))
        
        # 按文件名排序，确保时序一致性
        data_paths.sort()
        
        return data_paths
    
    def _group_sequences(self) -> List[List[str]]:
        """将连续的图像分组为时序序列"""
        sequence_groups = []
        
        for i in range(len(self.data_paths) - self.sequence_length + 1):
            sequence = self.data_paths[i:i + self.sequence_length]
            sequence_groups.append(sequence)
        
        return sequence_groups
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载单张图像"""
        try:
            # 使用PIL加载图像
            image = Image.open(image_path).convert('L')  # 转换为灰度图
            image = np.array(image)
            
            # 归一化到[0, 1]
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            
            # 转换为tensor: (H, W) -> (1, H, W)
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个默认的黑色图像
            return torch.zeros(1, 64, 64)
    
    def _extract_patch(self, image: torch.Tensor, center_x: int, center_y: int) -> torch.Tensor:
        """从图像中提取指定中心的patch"""
        H, W = image.shape[-2:]
        
        # 计算patch的边界
        half_size = self.patch_size // 2
        start_x = max(0, center_x - half_size)
        end_x = min(W, center_x + half_size)
        start_y = max(0, center_y - half_size)
        end_y = min(H, center_y + half_size)
        
        # 提取patch
        patch = image[..., start_y:end_y, start_x:end_x]
        
        # 如果patch大小不足，进行padding
        if patch.shape[-2] < self.patch_size or patch.shape[-1] < self.patch_size:
            pad_h = max(0, self.patch_size - patch.shape[-2])
            pad_w = max(0, self.patch_size - patch.shape[-1])
            patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
        
        return patch
    
    def _augment_data(self, image: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        if not self.augment:
            return image
        
        # 随机水平翻转
        if random.random() > 0.5:
            image = torch.flip(image, [-1])
        
        # 随机垂直翻转
        if random.random() > 0.5:
            image = torch.flip(image, [-2])
        
        # 随机旋转90度
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k, [-2, -1])
        
        return image
    
    def __len__(self) -> int:
        """返回序列组的数量"""
        return len(self.sequence_groups)
    
    def __getitem__(self, idx: int) -> dict:
        """获取一个数据样本"""
        sequence_paths = self.sequence_groups[idx]
        
        # 加载时序图像
        sequence_images = []
        for path in sequence_paths:
            image = self._load_image(path)
            sequence_images.append(image)
        
        # 选择中间帧作为目标高分辨率图像
        target_idx = self.sequence_length // 2
        target_hr = sequence_images[target_idx]
        
        # 生成低分辨率图像（通过下采样）
        target_lr = F.interpolate(
            target_hr.unsqueeze(0), 
            scale_factor=1/self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 数据增强
        if self.augment:
            target_hr = self._augment_data(target_hr)
            target_lr = self._augment_data(target_lr)
            sequence_images = [self._augment_data(img) for img in sequence_images]
        
        # 随机选择patch位置
        H, W = target_hr.shape[-2:]
        center_x = random.randint(self.patch_size//2, W - self.patch_size//2)
        center_y = random.randint(self.patch_size//2, H - self.patch_size//2)
        
        # 提取patch
        hr_patch = self._extract_patch(target_hr, center_x, center_y)
        lr_patch = self._extract_patch(target_lr, center_x, center_y)
        
        # 提取时序patch
        sequence_patches = []
        for img in sequence_images:
            patch = self._extract_patch(img, center_x, center_y)
            sequence_patches.append(patch)
        
        # 拼接时序patch用于时间判别器: (C*3, H, W)
        sequence_combined = torch.cat(sequence_patches, dim=0)
        
        return {
            'lr': lr_patch,                    # 低分辨率patch: (1, H, W)
            'hr': hr_patch,                    # 高分辨率patch: (1, H, W)
            'sequence': sequence_combined,      # 时序序列: (3, H, W)
            'sequence_paths': sequence_paths,   # 图像路径（用于调试）
            'patch_center': (center_x, center_y)  # patch中心位置
        }

class DigitalTyphoonDataLoader:
    """数据加载器工厂类"""
    
    @staticmethod
    def create_dataloaders(data_root: str,
                          batch_size: int = 16,
                          scale_factor: int = 4,
                          patch_size: int = 64,
                          sequence_length: int = 3,
                          num_workers: int = 4,
                          augment: bool = True):
        """
        创建训练、验证和测试数据加载器
        
        Args:
            data_root: 数据根目录
            batch_size: 批次大小
            scale_factor: 超分辨率倍数
            patch_size: 训练时的图像块大小
            sequence_length: 时序序列长度
            num_workers: 数据加载的工作进程数
            augment: 是否使用数据增强
        """
        from torch.utils.data import DataLoader
        
        # 创建数据集
        train_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='train',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=augment
        )
        
        val_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='val',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=False  # 验证时不使用数据增强
        )
        
        test_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='test',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=False  # 测试时不使用数据增强
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        return train_loader, val_loader, test_loader

# 使用示例
if __name__ == "__main__":
    # 创建数据集
    dataset = DigitalTyphoonDataset(
        data_root="./data/digital_typhoon",
        split="train",
        scale_factor=4,
        patch_size=64,
        sequence_length=3
    )
    
    # 获取一个样本
    sample = dataset[0]
    print(f"LR shape: {sample['lr'].shape}")
    print(f"HR shape: {sample['hr'].shape}")
    print(f"Sequence shape: {sample['sequence'].shape}")
    print(f"Patch center: {sample['patch_center']}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = DigitalTyphoonDataLoader.create_dataloaders(
        data_root="./data/digital_typhoon",
        batch_size=8,
        scale_factor=4,
        patch_size=64,
        sequence_length=3
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


