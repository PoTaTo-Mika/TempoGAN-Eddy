## 记录

这个digital typhoon的数据集是data/WP/image - metadata - aux_data.csv - metadata.json,其中的image下的组成为{年份}+{西太平洋台风编号}，每个编号的台风底下都有许多个h5文件，然后h5文件的命名格式应该是YYYYMMDDTT-YYYY{Typhoon ID(当年第N号台风)}-HMW8-1.H5

然后我们使用 test_data.py，拆解随机一个h5文件，得到：

```
root@autodl-container-2337448a2f-b25a3a94:~/autodl-tmp# python TempoGAN-Eddy/dataset/test_data.py data/WP/image/202107/2021071900-202107-HMW8-1.h5 
🌪️  Digital Typhoon H5文件结构查看器
==================================================

============================================================
文件路径: data/WP/image/202107/2021071900-202107-HMW8-1.h5
文件大小: 0.43 MB
============================================================
📁 / (Group)
   包含 1 个项目:
  📊 Infrared (Dataset)
     形状: (512, 512)
     数据类型: float64
     压缩: gzip
     样本数据 (前3x3):
     [[277.10538462 276.37615385 276.49769231]
 [277.22692308 276.01153846 276.25461538]
 [276.61923077 277.22692308 276.13307692]]
     属性: {'long_name': b'Brightness temperature', 'units': b'K (kelvin)'}
```