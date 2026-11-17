#========数据预处理===========================================
# 定义模块的公开接口，仅暴露ImageDataset类
__all__ = ["ImageDataset"]

from PIL import Image  # 导入PIL库中的Image模块，用于图像处理
import os  # 导入os库，用于处理文件和目录路径

from sympy.core import random
from torch.utils.data import Dataset, random_split  # 从PyTorch的工具库中导入Dataset类，用于自定义数据集
import denoising_config
from denoising_config import *
from common.utils import sorted_alphanumeric
import torch
import torchvision.transforms as T  # 用于图像预处理
import numpy as np



class ImageDataset(Dataset):  # 定义一个名为ImageDataset的类，继承自PyTorch的Dataset类
    def __init__(self, main_dir, transform=None):  # 定义类的构造函数，接受两个参数：main_dir（主目录路径）和transform（图像预处理操作，默认为None）
        self.main_dir = main_dir  # 将主目录路径保存为类的属性
        self.transform = transform  # 将图像预处理操作保存为类的属性
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))  # 获取所有图像文件名并按字母数字顺序排序

    def __len__(self):  # 定义__len__方法，返回数据集中图像的数量
        return len(self.all_imgs)  # 返回主目录下图像文件的数量

    def __getitem__(self, idx):  # 定义__getitem__方法，根据索引idx获取图像
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])  # 根据索引idx构建图像的完整路径
        image = Image.open(img_loc).convert("RGB")  # 使用PIL库打开图像并将其转换为RGB格式

        if self.transform is not None:  # 如果定义了图像预处理操作
            tensor_image = self.transform(image)  # 对图像进行预处理，并将其转换为张量
        else:
            # 若无变换，抛出异常提示必须提供预处理
            raise ValueError("transform参数不能为None，需指定预处理方法")

        ## 向输入图像添加随机噪声
        # 生成与 tensor_image 形状相同的随机噪声，乘以噪声因子 noise_factor
        noisy_imgs = tensor_image + denoising_config.NOISE_FACTOR * torch.randn(*tensor_image.shape)
        # 将图像像素值裁剪到 [0, 1] 范围内，避免超出有效范围
        noisy_imgs = torch.clip(noisy_imgs, 0., 1.)

        return noisy_imgs, tensor_image  # 返回预处理后的图像张量（将噪声图片作为输入，将原图作为目标）

#创建并切分数据集  的方法
def create_dataset():
    #定义转换操作
    transform = T.Compose([
        T.Resize((64, 64)),  # 调整图像大小为 256x256
        T.ToTensor(),  # 将图像转换为 PyTorch 张量
    ])
    dataset = ImageDataset(main_dir=IMG_PATH, transform=transform)
    #划分数据集
    train_dataset,val_dataset,test_dataset = random_split(dataset,[TRAIN_RATIO,VAL_RATIO,TEST_RATIO])
    return train_dataset,val_dataset,test_dataset

if __name__ == "__main__":
    dataset = ImageDataset(IMG_PATH)
    print(len(dataset))
    print(dataset.all_imgs)