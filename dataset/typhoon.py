# Dataset for digital typhoon
import torchvision
import torch

# 这个数据集要考虑"密度场"和"速度场"
# 密度场就假设按照云图上面的强度
# 速度差我们把速度矢量视作时间刻的速度

