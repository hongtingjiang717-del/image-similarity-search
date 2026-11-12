# 在公共图片数据集上增加“加噪声”操作，生成用于学习的图片对。
# 返回顺序固定为 (noisy, clean)：带噪声图片是输入，干净原图是希望模型恢复的目标。
# 像素已缩放到 [0,1]；高斯噪声可正可负，最后 clamp 把结果裁回合法范围。
# 训练噪声每次变化；验证传入 noise_seed 后，对同一索引固定噪声，便于比较不同轮次。

"""干净图像加高斯噪声，返回（带噪输入，干净目标）。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 common.data 导入 ImageFolderDataset, sorted_alphanumeric；函数/类在对应文件实现，本文件直接复用。
from common.data import ImageFolderDataset, sorted_alphanumeric
# 包内相对导入：前面的点表示当前包；从 本包 取出 denoising_config as config，避免误找同名顶层模块。
from . import denoising_config as config

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["ImageDataset"]

# 定义 ImageDataset 数据集类并继承父类能力；只重写与当前任务不同的部分。
class ImageDataset(ImageFolderDataset):
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self, main_dir, transform=None, noise_factor=config.NOISE_FACTOR, noise_seed=None):
        # 调用父类的数据集初始化，复用图片枚举和预处理逻辑，避免重复实现。
        super().__init__(main_dir, transform)
        # 检查数值参数的合法范围，不允许负噪声强度或非正的批次/结果数量。
        if noise_factor < 0:
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("noise_factor 不能为负数")
        # 保存噪声强度，供后续每次取图片时使用。
        self.noise_factor = noise_factor
        # 保存可选噪声种子；None 表示每次使用正常随机噪声，整数表示固定验证噪声。
        self.noise_seed = noise_seed

    # Python 执行 dataset[idx] 时自动调用这个方法；DataLoader 也通过它逐个取样本。
    def __getitem__(self, idx):
        # 读取当前索引对应的干净原图，作为去噪训练的目标。
        clean = self.load_image(idx)
        # 训练使用随机噪声；验证按图片索引固定噪声，避免每轮验证条件变化。
        # 先不创建专用发生器，默认使用 PyTorch 当前随机源。
        generator = None
        # is not None 判断调用者是否提供固定种子；None 表示没有指定。
        if self.noise_seed is not None:
            # 为当前图片创建固定种子的发生器；同一索引可复现噪声，不同索引使用不同种子。
            generator = torch.Generator().manual_seed(self.noise_seed + idx)
        # 生成与干净图片形状、数值类型、设备一致的标准正态噪声；均值 0，标准差 1。
        noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        # 原图加缩放后的噪声；clamp 将小于 0 的值设为 0、大于 1 的值设为 1。
        noisy = (clean + self.noise_factor * noise).clamp(0, 1)
        # 按（带噪输入，干净目标）顺序返回；训练循环会用两个变量接住这两个对象。
        return noisy, clean
