# 【初学者阅读说明】
# 本任务通过“重建原图”学习特征，因此输入和目标都是同一张干净图片。
# 真正的读图代码继承自 common.data.ImageFolderDataset；这里的 pass 不是遗漏实现。

"""重建训练返回（原图，原图）；文件名与嵌入始终使用相同顺序。"""
# 从 common.data 导入 ImageFolderDataset, sorted_alphanumeric；函数/类在对应文件实现，本文件直接复用。
from common.data import ImageFolderDataset, sorted_alphanumeric

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["ImageDataset"]

# 定义 ImageDataset 数据集类并继承父类能力；只重写与当前任务不同的部分。
class ImageDataset(ImageFolderDataset):
    # pass 是合法的空语句；这里表示继承父类实现或按已有约定不再执行额外动作，不是网络计算。
    pass
