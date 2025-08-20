#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Typhoon数据集H5文件结构查看器
用于查看指定路径下H5文件的内部结构和数据信息
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path
import json


def print_h5_structure(file_path, max_items=10, show_data_sample=True):
    """
    打印H5文件的结构信息
    
    Args:
        file_path (str): H5文件路径
        max_items (int): 显示的最大项目数量
        show_data_sample (bool): 是否显示数据样本
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"\n{'='*60}")
            print(f"文件路径: {file_path}")
            print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"{'='*60}")
            
            def print_group_info(name, obj, level=0):
                indent = "  " * level
                
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📁 {name}/ (Group)")
                    # 显示组的属性
                    if obj.attrs:
                        print(f"{indent}   属性: {dict(obj.attrs)}")
                    
                    # 限制显示的项目数量
                    items = list(obj.keys())
                    if len(items) > max_items:
                        print(f"{indent}   包含 {len(items)} 个项目 (显示前{max_items}个):")
                        items = items[:max_items]
                    else:
                        print(f"{indent}   包含 {len(items)} 个项目:")
                    
                    for item_name in items:
                        item = obj[item_name]
                        if isinstance(item, h5py.Dataset):
                            print_dataset_info(item_name, item, level + 1)
                        else:
                            print_group_info(item_name, item, level + 1)
                            
                elif isinstance(obj, h5py.Dataset):
                    print_dataset_info(name, obj, level)
            
            def print_dataset_info(name, dataset, level):
                indent = "  " * level
                print(f"{indent}📊 {name} (Dataset)")
                print(f"{indent}   形状: {dataset.shape}")
                print(f"{indent}   数据类型: {dataset.dtype}")
                print(f"{indent}   压缩: {dataset.compression}")
                
                # 显示数据样本
                if show_data_sample and dataset.size > 0:
                    try:
                        if dataset.ndim == 1:
                            sample_size = min(5, dataset.shape[0])
                            sample = dataset[:sample_size]
                            print(f"{indent}   样本数据: {sample}")
                        elif dataset.ndim == 2:
                            sample_size = min(3, min(dataset.shape[0], dataset.shape[1]))
                            sample = dataset[:sample_size, :sample_size]
                            print(f"{indent}   样本数据 (前{sample_size}x{sample_size}):")
                            print(f"{indent}   {sample}")
                        elif dataset.ndim == 3:
                            sample_size = min(2, min(dataset.shape[0], dataset.shape[1], dataset.shape[2]))
                            sample = dataset[:sample_size, :sample_size, :sample_size]
                            print(f"{indent}   样本数据 (前{sample_size}x{sample_size}x{sample_size}):")
                            print(f"{indent}   {sample}")
                        else:
                            print(f"{indent}   样本数据: 维度过多，跳过显示")
                    except Exception as e:
                        print(f"{indent}   样本数据: 无法读取 ({e})")
                
                # 显示数据集属性
                if dataset.attrs:
                    print(f"{indent}   属性: {dict(dataset.attrs)}")
            
            # 遍历文件结构
            print_group_info("", f)
            
    except Exception as e:
        print(f"❌ 无法读取文件 {file_path}: {e}")


def find_h5_files(directory, max_files=5):
    """
    在指定目录下查找H5文件
    
    Args:
        directory (str): 搜索目录
        max_files (int): 最大显示文件数量
    
    Returns:
        list: H5文件路径列表
    """
    h5_files = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.h5') or file.endswith('.hdf5'):
                    h5_files.append(os.path.join(root, file))
                    if len(h5_files) >= max_files:
                        break
            if len(h5_files) >= max_files:
                break
    except Exception as e:
        print(f"❌ 搜索目录时出错: {e}")
    
    return h5_files


def main():
    """主函数"""
    print("🌪️  Digital Typhoon H5文件结构查看器")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python test_data.py <h5文件路径>")
        print("  python test_data.py <目录路径> --search")
        print("  python test_data.py --help")
        print("\n示例:")
        print("  python test_data.py data/WP/image/202101/202101.h5")
        print("  python test_data.py data/WP/image/202101 --search")
        return
    
    if sys.argv[1] == "--help":
        print("帮助信息:")
        print("  --search: 在指定目录下搜索H5文件")
        print("  --max-items <n>: 限制显示的最大项目数量 (默认: 10)")
        print("  --no-sample: 不显示数据样本")
        return
    
    # 解析参数
    file_path = sys.argv[1]
    search_mode = "--search" in sys.argv
    max_items = 10
    show_sample = True
    
    for i, arg in enumerate(sys.argv):
        if arg == "--max-items" and i + 1 < len(sys.argv):
            try:
                max_items = int(sys.argv[i + 1])
            except ValueError:
                print("❌ --max-items 参数必须是数字")
                return
        elif arg == "--no-sample":
            show_sample = False
    
    if search_mode:
        # 搜索模式
        if not os.path.isdir(file_path):
            print(f"❌ 目录不存在: {file_path}")
            return
        
        print(f"🔍 在目录 {file_path} 中搜索H5文件...")
        h5_files = find_h5_files(file_path)
        
        if not h5_files:
            print("❌ 未找到H5文件")
            return
        
        print(f"📁 找到 {len(h5_files)} 个H5文件:")
        for i, h5_file in enumerate(h5_files, 1):
            print(f"  {i}. {h5_file}")
        
        if len(h5_files) == 1:
            print(f"\n📊 显示唯一找到的文件结构:")
            print_h5_structure(h5_files[0], max_items, show_sample)
        else:
            print(f"\n💡 使用 'python test_data.py <文件路径>' 查看特定文件结构")
    
    else:
        # 单文件模式
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            return
        
        if not (file_path.endswith('.h5') or file_path.endswith('.hdf5')):
            print(f"❌ 不是H5文件: {file_path}")
            return
        
        print_h5_structure(file_path, max_items, show_sample)


if __name__ == "__main__":
    main()
