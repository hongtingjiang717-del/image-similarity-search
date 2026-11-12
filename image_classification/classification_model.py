# 【初学者阅读说明】
# 两层卷积学习图片特征，再用全连接层输出五个类别的分数。
# 输入 [B,3,64,64] → 第一组 [B,8,32,32] → 第二组 [B,16,16,16]。
# 展平后 [B,4096] → 全连接输出 [B,5]；这五个数叫 logits，还不是概率。
# 训练用 CrossEntropyLoss 接收 logits；展示时才用 softmax 转成总和为 1 的五个数。

"""教材两层 CNN 五分类模型，输出原始 logits，配合 CrossEntropyLoss。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 导入 torch.nn，使用张量、神经网络计算及自动求导；as nn 给它起较短别名
import torch.nn as nn
# 导入 torch.nn.functional，使用张量、神经网络计算及自动求导；as F 给它起较短别名
import torch.nn.functional as F

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["Classifier"]

# 定义 Classifier 网络类，继承 nn.Module 以便 PyTorch 自动登记子层、参数及训练模式。
class Classifier(nn.Module):
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self, num_classes=5):
        # 调用父类 nn.Module 的初始化，建立参数与子层管理机制；先完成它，才能登记 self.conv1 等网络层。
        super().__init__()
        # 建立二维卷积层：输入通道 3、输出通道 8、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.pool = nn.MaxPool2d(2, 2)
        # 建立二维卷积层：输入通道 8、输出通道 16、卷积核 3；本项目常用 padding=1 保持高宽。
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # 建立全连接层：每张图的 16×16×16=4096 个特征分别连接到各类别输出；参数由训练学习。
        self.fc1 = nn.Linear(16 * 16 * 16, num_classes)

    # 定义模型前向计算；外部写 model(x) 时，PyTorch 的调用机制会执行这里的运算。
    # x 是张量，后面的 x = ... 表示把中间结果继续传给下一层，不是反向传播。
    def forward(self, x):
        # 检查输入/输出的维度与大小是否符合当前网络约定；不匹配时提前提示，而不是让矩阵运算难懂地报错。
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("分类输入应为 [B,3,64,64]")
        # 依次执行卷积、ReLU、最大池化；通道按卷积层变化，高宽经池化各缩小一半。
        x = self.pool(F.relu(self.conv1(x)))
        # 依次执行卷积、ReLU、最大池化；通道按卷积层变化，高宽经池化各缩小一半。
        x = self.pool(F.relu(self.conv2(x)))
        # flatten(1) 保留批次轴，把其余轴合并成 4096 个数，再经全连接得到五类原始分数。
        return self.fc1(x.flatten(1))

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
    print(Classifier()(torch.rand(1, 3, 64, 64)).shape)
