import os
import json
import time
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# 导入自定义模块
from model.tempogan import Generator, DiscriminatorS, DiscriminatorT
from dataset.typhoon import DigitalTyphoonDataLoader

class TempoGANTrainer:
    """TempoGAN训练器"""
    
    def __init__(self, config_path: str):
        """初始化训练器"""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['device'])
        
        # 创建必要的目录
        self._create_directories()
        
        # 设置日志
        self._setup_logging()
        
        # 初始化模型
        self._init_models()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 初始化损失函数
        self._init_loss_functions()
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(self.config['training']['log_dir'])
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        
        self.logger.info("TempoGAN训练器初始化完成")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    
    def _create_directories(self):
        """创建必要的目录"""
        Path(self.config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['training']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('TempoGAN')
        self.logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_file = os.path.join(self.config['training']['log_dir'], 'training.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _init_models(self):
        """初始化模型"""
        model_config = self.config['model']
        
        # 生成器
        self.generator = Generator(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            n_residual_blocks=model_config['n_residual_blocks'],
            upsample_factor=model_config['upsample_factor']
        ).to(self.device)
        
        # 空间判别器
        self.discriminator_s = DiscriminatorS(
            in_channels=model_config['in_channels']
        ).to(self.device)
        
        # 时间判别器
        self.discriminator_t = DiscriminatorT(
            in_channels=model_config['in_channels']
        ).to(self.device)
        
        self.logger.info(f"模型初始化完成，设备: {self.device}")
        self.logger.info(f"生成器参数数量: {sum(p.numel() for p in self.generator.parameters())}")
        self.logger.info(f"空间判别器参数数量: {sum(p.numel() for p in self.discriminator_s.parameters())}")
        self.logger.info(f"时间判别器参数数量: {sum(p.numel() for p in self.discriminator_t.parameters())}")
    
    def _init_optimizers(self):
        """初始化优化器"""
        training_config = self.config['training']
        
        # 生成器优化器
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=training_config['lr_g'],
            betas=(training_config['beta1'], training_config['beta2'])
        )
        
        # 判别器优化器
        self.optimizer_d = optim.Adam(
            list(self.discriminator_s.parameters()) + list(self.discriminator_t.parameters()),
            lr=training_config['lr_d'],
            betas=(training_config['beta1'], training_config['beta2'])
        )
        
        self.logger.info("优化器初始化完成")
    
    def _init_loss_functions(self):
        """初始化损失函数"""
        # 内容损失（L1损失）
        self.content_criterion = nn.L1Loss()
        
        # 对抗性损失（二元交叉熵）
        self.adversarial_criterion = nn.BCELoss()
        
        self.logger.info("损失函数初始化完成")
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        training_config = self.config['training']
        model_config = self.config['model']
        
        self.train_loader, self.val_loader, self.test_loader = DigitalTyphoonDataLoader.create_dataloaders(
            data_root=training_config['data_root'],
            batch_size=training_config['batch_size'],
            scale_factor=model_config['scale_factor'],
            patch_size=training_config['patch_size'],
            sequence_length=training_config['sequence_length'],
            num_workers=training_config['num_workers'],
            augment=True
        )
        
        self.logger.info(f"数据加载器初始化完成")
        self.logger.info(f"训练集批次数: {len(self.train_loader)}")
        self.logger.info(f"验证集批次数: {len(self.val_loader)}")
        self.logger.info(f"测试集批次数: {len(self.test_loader)}")
    
    def _train_discriminators(self, batch_data):
        """训练判别器"""
        # 准备数据
        lr_images = batch_data['lr'].to(self.device)  # (B, 1, H, W)
        hr_images = batch_data['hr'].to(self.device)  # (B, 1, H, W)
        sequence_images = batch_data['sequence'].to(self.device)  # (B, 3, H, W)
        
        batch_size = lr_images.size(0)
        
        # 创建真实和虚假标签
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # 生成假图像（不计算梯度）
        with torch.no_grad():
            sr_images = self.generator(lr_images)
        
        # 训练空间判别器
        self.optimizer_d.zero_grad()
        
        # 真实图像的判别结果
        real_validity_s = self.discriminator_s(hr_images, lr_images)
        d_real_loss_s = self.adversarial_criterion(real_validity_s, real_labels)
        
        # 假图像的判别结果
        fake_validity_s = self.discriminator_s(sr_images.detach(), lr_images)
        d_fake_loss_s = self.adversarial_criterion(fake_validity_s, fake_labels)
        
        # 空间判别器总损失
        d_loss_s = d_real_loss_s + d_fake_loss_s
        
        # 训练时间判别器
        # 真实序列的判别结果
        real_validity_t = self.discriminator_t(sequence_images)
        d_real_loss_t = self.adversarial_criterion(real_validity_t, real_labels)
        
        # 假序列的判别结果（需要重新生成三帧）
        with torch.no_grad():
            # 从sequence中提取三帧对应的低分辨率图像
            lr_prev = lr_images  # 简化处理，实际应该有三帧
            lr_curr = lr_images
            lr_next = lr_images
            
            sr_prev = self.generator(lr_prev)
            sr_curr = self.generator(lr_curr)
            sr_next = self.generator(lr_next)
            
            # 拼接三帧假图像
            fake_sequence = torch.cat([sr_prev, sr_curr, sr_next], dim=1)
        
        fake_validity_t = self.discriminator_t(fake_sequence.detach())
        d_fake_loss_t = self.adversarial_criterion(fake_validity_t, fake_labels)
        
        # 时间判别器总损失
        d_loss_t = d_real_loss_t + d_fake_loss_t
        
        # 总判别器损失
        d_loss = d_loss_s + d_loss_t
        
        # 反向传播
        d_loss.backward()
        self.optimizer_d.step()
        
        return {
            'd_loss_s': d_loss_s.item(),
            'd_loss_t': d_loss_t.item(),
            'd_loss': d_loss.item()
        }
    
    def _train_generator(self, batch_data):
        """训练生成器"""
        # 准备数据
        lr_images = batch_data['lr'].to(self.device)
        hr_images = batch_data['hr'].to(self.device)
        sequence_images = batch_data['sequence'].to(self.device)
        
        batch_size = lr_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        
        # 生成假图像
        sr_images = self.generator(lr_images)
        
        # 计算生成器损失
        self.optimizer_g.zero_grad()
        
        # 1. 内容损失（L1损失）
        content_loss = self.content_criterion(sr_images, hr_images)
        
        # 2. 对抗性损失 - 来自空间判别器
        fake_validity_s = self.discriminator_s(sr_images, lr_images)
        adversarial_loss_s = self.adversarial_criterion(fake_validity_s, real_labels)
        
        # 3. 对抗性损失 - 来自时间判别器
        # 生成三帧假图像
        lr_prev = lr_images  # 简化处理
        lr_curr = lr_images
        lr_next = lr_images
        
        sr_prev = self.generator(lr_prev)
        sr_curr = self.generator(lr_curr)
        sr_next = self.generator(lr_next)
        
        fake_sequence = torch.cat([sr_prev, sr_curr, sr_next], dim=1)
        fake_validity_t = self.discriminator_t(fake_sequence)
        adversarial_loss_t = self.adversarial_criterion(fake_validity_t, real_labels)
        
        # 总生成器损失
        training_config = self.config['training']
        total_g_loss = (
            training_config['lambda_content'] * content_loss +
            training_config['lambda_adv_s'] * adversarial_loss_s +
            training_config['lambda_adv_t'] * adversarial_loss_t
        )
        
        # 反向传播
        total_g_loss.backward()
        self.optimizer_g.step()
        
        return {
            'content_loss': content_loss.item(),
            'adversarial_loss_s': adversarial_loss_s.item(),
            'adversarial_loss_t': adversarial_loss_t.item(),
            'g_loss': total_g_loss.item()
        }
    
    def _validate(self):
        """验证模型"""
        self.generator.eval()
        self.discriminator_s.eval()
        self.discriminator_t.eval()
        
        total_content_loss = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                lr_images = batch_data['lr'].to(self.device)
                hr_images = batch_data['hr'].to(self.device)
                
                # 生成超分辨率图像
                sr_images = self.generator(lr_images)
                
                # 计算内容损失
                content_loss = self.content_criterion(sr_images, hr_images)
                total_content_loss += content_loss.item()
                
                # 计算PSNR
                sr_images_np = sr_images.cpu().numpy()
                hr_images_np = hr_images.cpu().numpy()
                
                for i in range(sr_images_np.shape[0]):
                    psnr = self._calculate_psnr(sr_images_np[i], hr_images_np[i])
                    total_psnr += psnr
                
                num_batches += 1
        
        # 恢复训练模式
        self.generator.train()
        self.discriminator_s.train()
        self.discriminator_t.train()
        
        avg_content_loss = total_content_loss / num_batches
        avg_psnr = total_psnr / (num_batches * self.config['training']['batch_size'])
        
        return {
            'val_content_loss': avg_content_loss,
            'val_psnr': avg_psnr
        }
    
    def _calculate_psnr(self, sr_img, hr_img):
        """计算PSNR"""
        # 确保图像值在[0, 1]范围内
        sr_img = np.clip(sr_img, 0, 1)
        hr_img = np.clip(hr_img, 0, 1)
        
        # 计算MSE
        mse = np.mean((sr_img - hr_img) ** 2)
        if mse == 0:
            return float('inf')
        
        # 计算PSNR
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_s_state_dict': self.discriminator_s.state_dict(),
            'discriminator_t_state_dict': self.discriminator_t.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(
            self.config['training']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存为best
        if is_best:
            best_path = os.path.join(
                self.config['training']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"最佳模型已保存: {best_path}")
        
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator_s.load_state_dict(checkpoint['discriminator_s_state_dict'])
        self.discriminator_t.load_state_dict(checkpoint['discriminator_t_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.logger.info(f"检查点已加载: {checkpoint_path}")
        self.logger.info(f"恢复训练: epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self, resume_from=None):
        """开始训练"""
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.logger.info("开始训练...")
        training_config = self.config['training']
        
        best_val_loss = float('inf')
        
        for epoch in range(self.current_epoch, training_config['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            epoch_losses = self._train_epoch()
            
            # 记录训练损失
            for loss_name, loss_value in epoch_losses.items():
                self.writer.add_scalar(f'Train/{loss_name}', loss_value, epoch)
            
            # 验证
            if (self.global_step + 1) % training_config['val_interval'] == 0:
                val_metrics = self._validate()
                for metric_name, metric_value in val_metrics.items():
                    self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
                
                # 检查是否是最佳模型
                if val_metrics['val_content_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_content_loss']
                    self._save_checkpoint(epoch, is_best=True)
            
            # 保存检查点
            if (epoch + 1) % training_config['save_interval'] == 0:
                self._save_checkpoint(epoch)
            
            # 记录学习率
            self.writer.add_scalar('Train/LR_Generator', 
                                 self.optimizer_g.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Train/LR_Discriminator', 
                                 self.optimizer_d.param_groups[0]['lr'], epoch)
            
            self.logger.info(f"Epoch {epoch+1}/{training_config['epochs']} 完成")
            self.logger.info(f"训练损失: {epoch_losses}")
        
        self.logger.info("训练完成！")
        self.writer.close()
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.generator.train()
        self.discriminator_s.train()
        self.discriminator_t.train()
        
        epoch_losses = {
            'd_loss_s': 0.0,
            'd_loss_t': 0.0,
            'd_loss': 0.0,
            'content_loss': 0.0,
            'adversarial_loss_s': 0.0,
            'adversarial_loss_t': 0.0,
            'g_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 训练判别器
            d_losses = self._train_discriminators(batch_data)
            
            # 训练生成器
            g_losses = self._train_generator(batch_data)
            
            # 累积损失
            for key in epoch_losses:
                if key in d_losses:
                    epoch_losses[key] += d_losses[key]
                elif key in g_losses:
                    epoch_losses[key] += g_losses[key]
            
            # 更新进度条
            progress_bar.set_postfix({
                'D_Loss': f"{d_losses['d_loss']:.4f}",
                'G_Loss': f"{g_losses['g_loss']:.4f}",
                'Content': f"{g_losses['content_loss']:.4f}"
            })
            
            # 记录到TensorBoard
            if (self.global_step + 1) % self.config['training']['log_interval'] == 0:
                for loss_name, loss_value in d_losses.items():
                    self.writer.add_scalar(f'Step/{loss_name}', loss_value, self.global_step)
                for loss_name, loss_value in g_losses.items():
                    self.writer.add_scalar(f'Step/{loss_name}', loss_value, self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses

def main():
    """主函数"""
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU训练")
    
    # 创建训练器
    trainer = TempoGANTrainer('configs/config.json')
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
