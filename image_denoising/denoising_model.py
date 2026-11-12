# 【初学者阅读说明】
# 本文件定义网络“结构”，创建对象时产生参数，参数要经过训练才有去噪能力。
# 编码：卷积提特征、最大池化缩小图像；解码：转置卷积扩大图像、普通卷积恢复 3 通道。
# 64×64 的空间尺寸：64→32→16→8→16→32→64；批次大小 B 始终不变。
# 编码各阶段通道：3→32→16→8；解码：8→8→16→32→3。
# ReLU 把负数变成 0；sigmoid 把输出压到 (0,1)，与目标图片的像素范围配合。

"""三层卷积去噪自编码器；默认 64×64 RGB，输入高宽须为 8 的倍数。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 导入 torch.nn，使用张量、神经网络计算及自动求导；as nn 给它起较短别名
import torch.nn as nn
# 导入 torch.nn.functional，使用张量、神经网络计算及自动求导；as F 给它起较短别名
import torch.nn.functional as F

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["ConvDenoiser"]

# 定义 ConvDenoiser 网络类，继承 nn.Module 以便 PyTorch 自动登记子层、参数及训练模式。
class ConvDenoiser(nn.Module):
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self):
        # 调用父类 nn.Module 的初始化，建立参数与子层管理机制；先完成它，才能登记 self.conv1 等网络层。
        super().__init__()
        # 建立二维卷积层：输入通道 3、输出通道 32、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 建立二维卷积层：输入通道 32、输出通道 16、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # 建立二维卷积层：输入通道 16、输出通道 8、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.pool = nn.MaxPool2d(2, 2)
        # 建立转置卷积层：通道 8→8；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)
        # 建立转置卷积层：通道 8→16；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        # 建立转置卷积层：通道 16→32；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        # 建立二维卷积层：输入通道 32、输出通道 3、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)

    # 定义模型前向计算；外部写 model(x) 时，PyTorch 的调用机制会执行这里的运算。
    # x 是张量，后面的 x = ... 表示把中间结果继续传给下一层，不是反向传播。
    def forward(self, x):
        # 检查输入/输出的维度与大小是否符合当前网络约定；不匹配时提前提示，而不是让矩阵运算难懂地报错。
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 8 or any(s % 8 for s in x.shape[-2:]):
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("去噪输入应为 [B,3,H,W]，H/W 为至少 8 的 8 倍数。")
        # 依次执行卷积、ReLU、最大池化；通道按卷积层变化，高宽经池化各缩小一半。
        x = self.pool(F.relu(self.conv1(x)))
        # 依次执行卷积、ReLU、最大池化；通道按卷积层变化，高宽经池化各缩小一半。
        x = self.pool(F.relu(self.conv2(x)))
        # 依次执行卷积、ReLU、最大池化；通道按卷积层变化，高宽经池化各缩小一半。
        x = self.pool(F.relu(self.conv3(x)))
        # 用转置卷积把高宽各放大两倍，再通过 ReLU；不是直接复制原图像素。
        x = F.relu(self.t_conv1(x))
        # 用转置卷积把高宽各放大两倍，再通过 ReLU；不是直接复制原图像素。
        x = F.relu(self.t_conv2(x))
        # 用转置卷积把高宽各放大两倍，再通过 ReLU；不是直接复制原图像素。
        x = F.relu(self.t_conv3(x))
        # 先把特征通道压回 RGB 的 3 通道，再用 sigmoid 限定输出像素范围。
        return self.conv_out(x).sigmoid()

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
    print(ConvDenoiser()(torch.rand(1, 3, 64, 64)).shape)
