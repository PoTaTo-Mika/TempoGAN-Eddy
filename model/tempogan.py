import torch
import torch.nn as nn
import torch.nn.functional as F

############## 组件 ##############
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
    
class UpsampleBlock2d(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock2d, self).__init__()
        # 使用插值放大特征图的 H, W 维度
        self.upsample = nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=False)
        # 使用2d卷积进一步学习和优化特征
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.prelu(x)
        return x

############## 本体 ##############
class Generator(nn.Module):
    # TempoGAN的基本实现是用一个全卷积ResNet网络做生成器
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=8, upsample_factor=4):
        super(Generator, self).__init__()
        
        # 将输入数据映射到高维特征空间
        self.head = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=5, padding=2), nn.PReLU())

        # 堆叠多个残差块
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(n_residual_blocks)])
        
        # 身体的后处理层，用于与头部进行全局残差连接
        self.body_post_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64))

        # 连续进行两次 2x 上采样，最终实现 4x
        self.tail = nn.Sequential(
            UpsampleBlock2d(64, up_scale=2),
            UpsampleBlock2d(64, up_scale=2),
            nn.Conv2d(64, out_channels, kernel_size=5, padding=2),
            nn.Tanh() # 将输出归一化到 [-1, 1] 范围，这在GAN中很常见
        )

    def forward(self, x):
        # 头部输出，用于全局残差连接
        x_head = self.head(x)
        
        # 身体部分
        x_body = self.body(x_head)
        x_body = self.body_post_conv(x_body)
        
        # 全局残差连接 (非常重要，能稳定训练并提升性能)
        x = x_head + x_body
        
        # 尾部上采样
        x = self.tail(x)
        
        return x

############## 图像判别器 ##############
class DiscriminatorS(nn.Module):
    def __init__(self, in_channels=1):
        super(DiscriminatorS, self).__init__()
        input_channels = in_channels * 2

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # size: (256, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, high_res_image, low_res_image):
        # 将低分辨率图像上采样到高分辨率图像的尺寸
        upsample_low_resolution = nn.functional.interpolate(
            low_res_image, 
            size=high_res_image.shape[2:], mode='bilinear', 
            align_corners=False
        )

        # 拼接高分辨率和上采样后的低分辨率图像
        combined_input = torch.cat([high_res_image, 
                                    upsample_low_resolution], 
                                   dim=1)
        
        features = self.features(combined_input)
        features = torch.flatten(features, 1)
        validity = self.classifier(features)

        return validity

############## 时间判别器 #############
def multi_frame_align(frame_previous, frame_next, velocity_previous, velocity_next):

    device = frame_previous.device
    N, C, H, W = frame_previous.shape

    # 创建2D网格
    grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing='ij'
    )

    # 形状: [H, W] -> [1, H, W, 2] -> [N, H, W, 2]
    identity_grid = torch.stack((grid_w, grid_h), dim=2).unsqueeze(0).to(device)
    identity_grid = identity_grid.repeat(N, 1, 1, 1)

    # 将速度场调整为网格格式
    # velocity: (N, 2, H, W) -> (N, H, W, 2)
    flow_prev = identity_grid + velocity_previous.permute(0, 2, 3, 1)
    flow_next = identity_grid - velocity_next.permute(0, 2, 3, 1)

    # 使用grid_sample进行图像变形
    warped_frame_prev = F.grid_sample(frame_previous, flow_prev, mode='bilinear', padding_mode='border', align_corners=False)
    warped_frame_next = F.grid_sample(frame_next, flow_next, mode='bilinear', padding_mode='border', align_corners=False)

    return warped_frame_prev, warped_frame_next

class DiscriminatorT(nn.Module):
    def __init__(self, in_channels=1):
        super(DiscriminatorT,self).__init__()

        input_channels = in_channels * 3  # 三帧图像拼接

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frame_sequence):
        # 输入 frame_sequence: (N, C*3, H, W) - 三帧图像拼接
        features = self.features(frame_sequence)
        features_flat = torch.flatten(features, 1)
        validity = self.classifier(features_flat)
        return validity