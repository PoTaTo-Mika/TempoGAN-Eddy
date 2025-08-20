#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Digital Typhoon卫星图像可视化器
将H5文件中的红外数据渲染成人类可读的图像
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_satellite_data(h5_file_path):
    """
    从H5文件加载卫星数据
    
    Args:
        h5_file_path (str): H5文件路径
    
    Returns:
        tuple: (数据数组, 属性信息)
    """
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # 获取红外数据
            infrared_data = f['Infrared'][:]
            
            # 获取属性信息
            attrs = dict(f['Infrared'].attrs)
            
            print(f"✅ 成功加载数据:")
            print(f"   数据形状: {infrared_data.shape}")
            print(f"   数据类型: {infrared_data.dtype}")
            print(f"   数据范围: {infrared_data.min():.2f} - {infrared_data.max():.2f}")
            print(f"   属性: {attrs}")
            
            return infrared_data, attrs
            
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return None, None

def save_satellite_images(data, output_dir, base_filename):
    """
    保存卫星图像为独立的图片文件
    
    Args:
        data (np.ndarray): 图像数据
        output_dir (str): 输出目录
        base_filename (str): 基础文件名（不含扩展名）
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 保存彩色图像 (Viridis配色)
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(data, cmap='viridis', aspect='equal')
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)
    
    color_path = os.path.join(output_dir, f"{base_filename}_color.png")
    plt.savefig(color_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"💾 彩色图像已保存: {color_path}")
    
    # 2. 保存灰度图像
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(data, cmap='gray', aspect='equal')
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout(pad=0)
    
    gray_path = os.path.join(output_dir, f"{base_filename}_gray.png")
    plt.savefig(gray_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"💾 灰度图像已保存: {gray_path}")
    
    return color_path, gray_path

def analyze_data_statistics(data):
    """
    分析数据统计信息
    
    Args:
        data (np.ndarray): 图像数据
    """
    print("\n📊 数据统计分析:")
    print(f"   最小值: {data.min():.2f} K")
    print(f"   最大值: {data.max():.2f} K")
    print(f"   平均值: {data.mean():.2f} K")
    print(f"   标准差: {data.std():.2f} K")
    print(f"   中位数: {np.median(data):.2f} K")
    
    # 计算温度分布
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    print(f"   百分位数:")
    for p in percentiles:
        value = np.percentile(data, p)
        print(f"     {p}%: {value:.2f} K")

def main():
    """主函数"""
    print("🌪️  Digital Typhoon 卫星图像可视化器")
    print("=" * 50)
    
    # H5文件路径
    h5_file = "data/2022111112-202224-HMW8-1.h5"
    
    if not os.path.exists(h5_file):
        print(f"❌ 文件不存在: {h5_file}")
        return
    
    print(f"📁 正在处理文件: {h5_file}")
    
    # 加载数据
    data, attrs = load_satellite_data(h5_file)
    
    if data is None:
        return
    
    # 分析数据统计信息
    analyze_data_statistics(data)
    
    # 保存图像
    output_dir = "data/visualization"
    base_filename = h5_file.split('/')[-1].split('.')[0]
    
    color_path, gray_path = save_satellite_images(data, output_dir, base_filename)
    
    print(f"\n🎉 图像保存完成！")
    print(f"   原始数据: {h5_file}")
    print(f"   彩色图像: {color_path}")
    print(f"   灰度图像: {gray_path}")
    print(f"   图像尺寸: 512×512 像素")

if __name__ == "__main__":
    main()
