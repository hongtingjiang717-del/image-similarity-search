import pandas as pd
from common.utils import sorted_alphanumeric
from PIL import Image
import os
import torch
from torch.utils.data import Dataset,DataLoader,random_split
import torchvision.transforms as T
from classification_config import *

# 自定义数据集及标签
class ImageLabelDataset(Dataset):  # 定义一个名为ImageDataset的类，继承自PyTorch的Dataset类
    def __init__(self, main_dir, label_path, transform=None):  # 定义类的构造函数，接受两个参数：main_dir（主目录路径）和transform（图像预处理操作，默认为None）
        self.main_dir = main_dir  # 将主目录路径保存为类的属性
        self.transform = transform  # 将图像预处理操作保存为类的属性
        self.all_imgs = sorted_alphanumeric(os.listdir(main_dir))  # 获取所有图像文件名并按字母数字顺序排序
        labels = pd.read_csv(label_path) #读取标签
        self.label_dict = dict(zip(labels['id'], labels['target']))  #将标签保存成字典

    def __len__(self):  # 定义__len__方法，返回数据集中图像的数量
        return len(self.all_imgs)  # 返回主目录下图像文件的数量

    def __getitem__(self, idx):  # 定义__getitem__方法，根据索引idx获取图像
        #1.根据索引号找到文件名，读取图片数据
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])  # 根据索引idx构建图像的完整路径
        image = Image.open(img_loc).convert("RGB")  # 使用PIL库打开图像并将其转换为RGB格式

        #2.将原始图片转换成符合模型输入要求的张量，这就是重构图像的目标
        if self.transform is not None:  # 如果定义了图像预处理操作
            tensor_image = self.transform(image)  # 对图像进行预处理，并将其转换为张量
        else:
            # 若无变换，抛出异常提示必须提供预处理
            raise ValueError("transform参数不能为None，需指定预处理方法")

        #3.找到图片对应的标签
        label = self.label_dict[idx]

        #将输入和目标返回
        return tensor_image,label  # 返回预处理后的图像张量（将噪声图片作为输入，将原图作为目标）

#创建并切分数据集  的方法
def create_dataset():
    #定义转换操作
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),  # 调整图像大小为 256x256
        T.ToTensor(),  # 将图像转换为 PyTorch 张量
    ])
    dataset = ImageLabelDataset(main_dir=IMG_PATH,label_path=FASHION_LABELS_PATH, transform=transform)
    #划分数据集
    train_dataset,val_dataset,test_dataset = random_split(dataset,[TRAIN_RATIO,VAL_RATIO,TEST_RATIO])
    return train_dataset,val_dataset,test_dataset

if __name__ == "__main__":
    train_dataset,val_dataset,test_dataset = create_dataset()
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))