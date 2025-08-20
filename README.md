# TempoGAN-Eddy

基于TempoGAN的台风云图超分辨率重建项目，专门用于处理时序台风云图数据。

## 项目结构

```
TempoGAN-Eddy/
├── configs/
│   └── config.json          # 配置文件
├── dataset/
│   └── typhoon.py           # 台风数据集类
├── model/
│   └── tempogan.py          # TempoGAN模型定义
├── train.py                 # 训练脚本
├── requirements.txt          # 依赖包列表
└── README.md               # 项目说明
```

## 环境要求

- Python 3.7+
- PyTorch 1.9+
- CUDA 11.0+ (推荐用于GPU训练)
- 4090 GPU (推荐)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

将台风云图数据按以下结构组织：

```
data/
└── digital_typhoon/
    ├── train/           # 训练集
    │   ├── typhoon_001/
    │   │   ├── frame_001.png
    │   │   ├── frame_002.png
    │   │   └── ...
    │   └── typhoon_002/
    │       └── ...
    ├── val/             # 验证集
    └── test/            # 测试集
```

## 配置参数

在 `configs/config.json` 中配置训练参数：

```json
{
    "model": {
        "in_channels": 1,           # 输入通道数
        "out_channels": 1,          # 输出通道数
        "n_residual_blocks": 8,     # 残差块数量
        "upsample_factor": 4,       # 上采样倍数
        "scale_factor": 4           # 超分辨率倍数
    },
    "training": {
        "data_root": "./data/digital_typhoon",  # 数据根目录
        "batch_size": 16,           # 批次大小
        "patch_size": 64,           # 训练patch大小
        "sequence_length": 3,       # 时序序列长度
        "epochs": 100,              # 训练轮数
        "lr_g": 0.0001,            # 生成器学习率
        "lr_d": 0.0001,            # 判别器学习率
        "lambda_content": 1.0,      # 内容损失权重
        "lambda_adv_s": 0.001,      # 空间对抗损失权重
        "lambda_adv_t": 0.001      # 时间对抗损失权重
    }
}
```

## 开始训练

### 从头开始训练

```bash
python train.py
```

### 从检查点恢复训练

```python
# 在train.py中修改
trainer = TempoGANTrainer('configs/config.json')
trainer.train(resume_from='./checkpoints/checkpoint_epoch_50.pth')
```

## 训练流程

### 1. 准备阶段
- 数据加载器返回连续三帧图像序列
- 自动生成对应的低分辨率图像

### 2. 训练循环

#### 步骤A：训练判别器
- 固定生成器权重
- 训练空间判别器 (Ds)
- 训练时间判别器 (Dt)

#### 步骤B：训练生成器
- 固定判别器权重
- 计算内容损失 (L1损失)
- 计算对抗性损失 (来自Ds和Dt)

## 监控训练

使用TensorBoard监控训练过程：

```bash
tensorboard --logdir=./logs
```

## 模型保存

- 每10个epoch保存一次检查点
- 自动保存最佳验证模型
- 检查点包含所有模型状态和优化器状态

## 注意事项

1. **数据格式**：确保图像为灰度图，值范围[0, 1]
2. **内存管理**：根据GPU内存调整batch_size和patch_size
3. **时序一致性**：数据加载器会自动处理时序分组
4. **损失权重**：根据实际效果调整lambda参数

## 性能优化

- 使用多进程数据加载 (`num_workers`)
- 启用混合精度训练 (PyTorch AMP)
- 使用梯度累积处理大批次
- 定期验证和早停机制

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小batch_size
   - 减小patch_size
   - 减少num_workers

2. **训练不稳定**
   - 调整学习率
   - 检查损失权重
   - 验证数据质量

3. **收敛缓慢**
   - 增加判别器训练频率
   - 调整损失权重
   - 检查数据预处理

## 引用

如果本项目对您的研究有帮助，请引用相关论文：

```
@article{tempogan2020,
  title={TempoGAN: A Temporally Coherent, Spatially High-Resolution GAN for Video Super-Resolution},
  author={...},
  journal={...},
  year={2020}
}
```
