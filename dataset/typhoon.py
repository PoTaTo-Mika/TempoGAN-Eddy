# Dataset for digital typhoon H5 files
import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
from torch.utils.data import Dataset
import random
from typing import List, Tuple, Optional
import glob

class DigitalTyphoonDataset(Dataset):
    """
    Digital Typhoon数据集类，用于超分辨率任务
    支持时序图像处理，可以生成连续三帧用于时间判别器
    直接从H5文件读取Infrared数据
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 scale_factor: int = 4,
                 patch_size: int = 64,
                 sequence_length: int = 3,
                 augment: bool = True,
                 num_typhoons: int = 100):  # 新增：指定台风数量
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录 (如 data/WP/image/)
            split: 数据集分割 ('train', 'val', 'test')
            scale_factor: 超分辨率倍数
            patch_size: 训练时的图像块大小
            sequence_length: 时序序列长度（用于时间判别器）
            augment: 是否使用数据增强
            num_typhoons: 指定使用的台风数量
        """
        self.data_root = data_root
        self.split = split
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.sequence_length = sequence_length
        self.augment = augment
        self.num_typhoons = num_typhoons
        
        # 数据路径列表
        self.data_paths = self._load_data_paths()
        
        # 时序分组：将连续的图像分组
        self.sequence_groups = self._group_sequences()
        
        print(f"Loaded {len(self.data_paths)} H5 files for {split} split")
        print(f"Created {len(self.sequence_groups)} sequence groups")
    
    def _load_data_paths(self) -> List[str]:
        """加载H5文件路径 - 改进策略：按台风编号随机选择"""
        data_paths = []
        
        # 获取所有台风编号目录
        typhoon_dirs = []
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            if os.path.isdir(item_path) and item.startswith('20'):
                typhoon_dirs.append(item)
        
        # 按台风编号排序
        typhoon_dirs.sort()
        
        # 根据split确定使用的台风数量
        if self.split == 'train':
            # 训练集：使用大部分台风
            num_use = int(self.num_typhoons * 0.7)
            start_idx = 0
        elif self.split == 'val':
            # 验证集：使用部分台风
            num_use = int(self.num_typhoons * 0.15)
            start_idx = int(self.num_typhoons * 0.7)
        elif self.split == 'test':
            # 测试集：使用剩余台风
            num_use = int(self.num_typhoons * 0.15)
            start_idx = int(self.num_typhoons * 0.85)
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # 确保不超出范围
        if start_idx + num_use > len(typhoon_dirs):
            num_use = len(typhoon_dirs) - start_idx
        
        # 选择指定范围的台风目录
        selected_typhoons = typhoon_dirs[start_idx:start_idx + num_use]
        
        print(f"Split {self.split}: Using {len(selected_typhoons)} typhoons from {start_idx} to {start_idx + num_use}")
        
        # 遍历选中的台风目录下的所有H5文件
        for typhoon_dir in selected_typhoons:
            typhoon_path = os.path.join(self.data_root, typhoon_dir)
            if os.path.exists(typhoon_path):
                # 查找该台风下的所有H5文件
                h5_files = glob.glob(os.path.join(typhoon_path, "*.h5"))
                # 按文件名排序，确保时序一致性
                h5_files.sort()
                data_paths.extend(h5_files)
        
        if not data_paths:
            raise ValueError(f"No H5 files found for split {self.split} in {self.data_root}")
        
        return data_paths
    
    def _group_sequences(self) -> List[List[str]]:
        """将连续的图像分组为时序序列 - 修复：确保同一台风内的连续性"""
        sequence_groups = []
        
        # 按台风目录分组
        typhoon_groups = {}
        for path in self.data_paths:
            # 从路径中提取台风编号
            typhoon_id = os.path.basename(os.path.dirname(path))
            if typhoon_id not in typhoon_groups:
                typhoon_groups[typhoon_id] = []
            typhoon_groups[typhoon_id].append(path)
        
        # 为每个台风创建时序序列
        for typhoon_id, paths in typhoon_groups.items():
            # 确保路径按时间排序
            paths.sort()
            
            # 创建时序序列组
            for i in range(len(paths) - self.sequence_length + 1):
                sequence = paths[i:i + self.sequence_length]
                sequence_groups.append(sequence)
        
        return sequence_groups
    
    def _load_h5_data(self, h5_path: str) -> torch.Tensor:
        """从H5文件加载Infrared数据"""
        try:
            with h5py.File(h5_path, 'r') as f:
                # 读取Infrared数据集
                if 'Infrared' in f:
                    data = f['Infrared'][:]
                else:
                    # 如果没有Infrared，尝试读取根目录下的第一个数据集
                    first_key = list(f.keys())[0]
                    data = f[first_key][:]
                
                # 转换为float32
                data = data.astype(np.float32)
                
                # 归一化到[0, 1]范围
                # 根据log.md中的数据，这是亮度温度数据，单位是K
                # 假设温度范围在200-300K之间，归一化到[0,1]
                data_min = np.min(data)
                data_max = np.max(data)
                if data_max > data_min:
                    data = (data - data_min) / (data_max - data_min)
                
                # 转换为tensor: (H, W) -> (1, H, W)
                data = torch.from_numpy(data).float().unsqueeze(0)
                
                return data
                
        except Exception as e:
            print(f"Error loading H5 file {h5_path}: {e}")
            # 返回一个默认的黑色图像
            return torch.zeros(1, 512, 512)  # 根据log.md中的数据尺寸
    
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
        
        # 计算需要的padding
        current_h, current_w = patch.shape[-2:]
        pad_h = max(0, self.patch_size - current_h)
        pad_w = max(0, self.patch_size - current_w)
        
        # 如果patch大小不足，进行padding
        if pad_h > 0 or pad_w > 0:
            # 确保patch不为空再进行padding
            if patch.numel() > 0:
                # 使用reflect模式进行padding，如果失败则使用zeros模式
                try:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                except RuntimeError:
                    # 如果reflect模式失败，使用zeros模式
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                # 如果patch为空，创建一个全零patch
                patch = torch.zeros(image.shape[0], self.patch_size, self.patch_size, 
                                  dtype=image.dtype, device=image.device)
        
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
        """获取一个数据样本 - 修复：确保三帧是不同的图像"""
        sequence_paths = self.sequence_groups[idx]
        
        # 加载时序图像
        sequence_images = []
        for path in sequence_paths:
            image = self._load_h5_data(path)
            sequence_images.append(image)
        
        # 选择中间帧作为目标高分辨率图像
        target_idx = self.sequence_length // 2
        target_hr = sequence_images[target_idx]
        
        # 生成低分辨率图像（通过下采样）
        # 注意：这里应该先提取patch，再下采样，而不是先下采样再提取patch
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
        # 确保选择的中心点能够提取到完整的patch
        min_center_x = max(self.patch_size//2, 0)
        max_center_x = min(W - self.patch_size//2, W)
        min_center_y = max(self.patch_size//2, 0)
        max_center_y = min(H - self.patch_size//2, H)
        
        # 如果图像太小，无法提取patch，则使用图像中心
        if max_center_x <= min_center_x or max_center_y <= min_center_y:
            center_x = W // 2
            center_y = H // 2
        else:
            center_x = random.randint(min_center_x, max_center_x)
            center_y = random.randint(min_center_y, max_center_y)
        
        # 提取patch - 修复：先提取高分辨率patch，再生成对应的低分辨率patch
        hr_patch = self._extract_patch(target_hr, center_x, center_y)
        
        # 从高分辨率patch生成低分辨率patch
        lr_patch = F.interpolate(
            hr_patch.unsqueeze(0), 
            scale_factor=1/self.scale_factor, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # 提取时序patch - 修复：确保三帧是不同的patch
        sequence_patches = []
        for img in sequence_images:
            patch = self._extract_patch(img, center_x, center_y)
            sequence_patches.append(patch)
        
        # 拼接时序patch用于时间判别器: (C*3, H, W)
        sequence_combined = torch.cat(sequence_patches, dim=0)
        
        return {
            'lr': lr_patch,                    # 低分辨率patch: (1, H/4, W/4)
            'hr': hr_patch,                    # 高分辨率patch: (1, H, W)
            'sequence': sequence_combined,      # 时序序列: (3, H, W)
            'sequence_paths': sequence_paths,   # 图像路径（用于调试）
            'patch_center': (center_x, center_y),  # patch中心位置
            'lr_sequence': [F.interpolate(
                patch.unsqueeze(0), 
                scale_factor=1/self.scale_factor, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) for patch in sequence_patches]  # 新增：三帧低分辨率图像
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
                          augment: bool = True,
                          num_typhoons: int = 100):  # 新增：指定台风数量
        """
        创建训练、验证和测试数据加载器
        
        Args:
            data_root: 数据根目录 (如 data/WP/image/)
            batch_size: 批次大小
            scale_factor: 超分辨率倍数
            patch_size: 训练时的图像块大小
            sequence_length: 时序序列长度
            num_workers: 数据加载的工作进程数
            augment: 是否使用数据增强
            num_typhoons: 指定使用的台风数量
        """
        from torch.utils.data import DataLoader
        
        # 创建数据集
        train_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='train',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=augment,
            num_typhoons=num_typhoons
        )
        
        val_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='val',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=False,  # 验证时不使用数据增强
            num_typhoons=num_typhoons
        )
        
        test_dataset = DigitalTyphoonDataset(
            data_root=data_root,
            split='test',
            scale_factor=scale_factor,
            patch_size=patch_size,
            sequence_length=sequence_length,
            augment=False,  # 测试时不使用数据增强
            num_typhoons=num_typhoons
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
        data_root="../data/WP/image",  # 修改为实际的H5文件路径
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
        data_root="../data/WP/image",  # 修改为实际的H5文件路径
        batch_size=8,
        scale_factor=4,
        patch_size=64,
        sequence_length=3
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


