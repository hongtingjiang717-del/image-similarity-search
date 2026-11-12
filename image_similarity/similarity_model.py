# 【初学者阅读说明】
# 教材的六层自编码器：编码器把图片变为 512 个数，解码器用这些数重建图片。
# 编码空间尺寸：64→32→16→8→4→2→1；通道：3→16→32→64→128→256→512。
# 编码器最终得到 [B,512,1,1]，只去掉末尾两个 1，得到 [B,512]；B=1 也不会丢掉批次维度。
# 解码器把 [B,512] 补回空间维度，再连续放大六次，得到 [B,3,64,64]。
# 重建训练只是在学习一种图片表示；向量相近是否符合人的商品判断，要另外评价。

# 定义模块的公开接口，仅暴露ConvEncoder和ConvDecoder类
# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["ConvEncoder", "ConvDecoder"]
# 导入神经网络模块
# 导入 torch.nn，使用张量、神经网络计算及自动求导；as nn 给它起较短别名
import torch.nn as nn
# 定义 ConvEncoder 网络类，继承 nn.Module 以便 PyTorch 自动登记子层、参数及训练模式。
class ConvEncoder(nn.Module):
    """
    卷积编码器：通过卷积和下采样操作压缩图像为低维特征表示
    输入形状：(batch_size, 3, H, W)  # 假设输入为RGB图像
    输出形状：(batch_size, 512)，64×64 图像经过六次池化
    """
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self):
        # 调用父类 nn.Module 的初始化，建立参数与子层管理机制；先完成它，才能登记 self.conv1 等网络层。
        super().__init__()  # 调用父类构造函数初始化
        # 定义编码器各层（输入通道, 输出通道, 卷积核大小, 填充）
        # Block 1: 3 -> 16 channels
        # 建立二维卷积层：输入通道 3、输出通道 16、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))  # 保持尺寸的卷积
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu1 = nn.ReLU(inplace=True)  # 原地ReLU激活，节省内存
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool1 = nn.MaxPool2d((2, 2))  # 尺寸减半
        # Block 2: 16 -> 32 channels
        # 建立二维卷积层：输入通道 16、输出通道 32、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu2 = nn.ReLU(inplace=True)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool2 = nn.MaxPool2d((2, 2))
        # Block 3: 32 -> 64 channels
        # 建立二维卷积层：输入通道 32、输出通道 64、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv3 = nn.Conv2d(32, 64, (3, 3), padding=(1, 1))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu3 = nn.ReLU(inplace=True)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool3 = nn.MaxPool2d((2, 2))
        # Block 4: 64 -> 128 channels
        # 建立二维卷积层：输入通道 64、输出通道 128、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv4 = nn.Conv2d(64, 128, (3, 3), padding=(1, 1))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu4 = nn.ReLU(inplace=True)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool4 = nn.MaxPool2d((2, 2))
        # Block 5: 128 -> 256 channels
        # 建立二维卷积层：输入通道 128、输出通道 256、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv5 = nn.Conv2d(128, 256, (3, 3), padding=(1, 1))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu5 = nn.ReLU(inplace=True)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool5 = nn.MaxPool2d((2, 2))
        # Block 6: 256 -> 512 channels
        # 建立二维卷积层：输入通道 256、输出通道 512、卷积核 (3, 3)；本项目常用 padding=1 保持高宽。
        self.conv6 = nn.Conv2d(256, 512, (3, 3), padding=(1, 1))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu6 = nn.ReLU(inplace=True)
        # 建立最大池化层：每个小窗口取最大值；这里通常用 2×2 窗口把高和宽各减半，不改变通道数。
        self.maxpool6 = nn.MaxPool2d((2, 2))
    # 定义模型前向计算；外部写 model(x) 时，PyTorch 的调用机制会执行这里的运算。
    # x 是张量，后面的 x = ... 表示把中间结果继续传给下一层，不是反向传播。
    def forward(self, x):
        """前向传播：按顺序通过各层实现下采样"""
        # 检查输入/输出的维度与大小是否符合当前网络约定；不匹配时提前提示，而不是让矩阵运算难懂地报错。
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("编码器输入应为 [B,3,64,64]")
        # 初始输入为 (batch, 3, 64, 64)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv1(x)  # -> (batch, 16, 64, 64)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu1(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool1(x)  # -> (batch, 16, 32, 32)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv2(x)  # -> (batch, 32, 32, 32)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu2(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool2(x)  # -> (batch, 32, 16, 16)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv3(x)  # -> (batch, 64, 16, 16)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu3(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool3(x)  # -> (batch, 64, 8, 8)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv4(x)  # -> (batch, 128, 8, 8)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu4(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool4(x)  # -> (batch, 128, 4, 4)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv5(x)  # -> (batch, 256, 4, 4)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu5(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool5(x)  # -> (batch, 256, 2, 2)
        # 把上一层特征输入当前卷积层，按该层设置改变通道数并提取新特征。
        x = self.conv6(x)  # -> (batch, 512, 2, 2)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu6(x)
        # 执行当前阶段最大池化，高宽各减半；六次池化把 64×64 压到 1×1。
        x = self.maxpool6(x)  # -> (batch, 512, 1, 1)
        # 只去掉最后两个长度为 1 的空间轴：[B,512,1,1]→[B,512]，保留 B=1 的批次轴。
        x = x.squeeze(-1).squeeze(-1)
        # 将本网络/当前阶段的最终张量返回给调用方；本函数到这里结束。
        return x  # 输出压缩后的特征表示
# 定义 ConvDecoder 网络类，继承 nn.Module 以便 PyTorch 自动登记子层、参数及训练模式。
class ConvDecoder(nn.Module):
    """
    卷积解码器：通过转置卷积上采样，将低维特征重建为原始图像
    输入形状：(batch_size, 512)
    输出形状：(batch_size, 3, H, W)  # 恢复原始尺寸
    """
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self):
        # 调用父类 nn.Module 的初始化，建立参数与子层管理机制；先完成它，才能登记 self.conv1 等网络层。
        super().__init__()
        # 定义解码器各层（输入通道, 输出通道, 核大小, 步长）
        # Block 1: 512 -> 256 channels
        # 建立转置卷积层：通道 512→256；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv1 = nn.ConvTranspose2d(512, 256, (2, 2), stride=(2, 2))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu1 = nn.ReLU(inplace=True)
        # Block 2: 256 -> 128 channels
        # 建立转置卷积层：通道 256→128；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv2 = nn.ConvTranspose2d(256, 128, (2, 2), stride=(2, 2))  # 尺寸翻倍
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu2 = nn.ReLU(inplace=True)
        # Block 3: 128 -> 64 channels
        # 建立转置卷积层：通道 128→64；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv3 = nn.ConvTranspose2d(128, 64, (2, 2), stride=(2, 2))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu3 = nn.ReLU(inplace=True)
        # Block 4: 64 -> 32 channels
        # 建立转置卷积层：通道 64→32；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv4 = nn.ConvTranspose2d(64, 32, (2, 2), stride=(2, 2))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu4 = nn.ReLU(inplace=True)
        # Block 5: 32 -> 16 channels
        # 建立转置卷积层：通道 32→16；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv5 = nn.ConvTranspose2d(32, 16, (2, 2), stride=(2, 2))
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu5 = nn.ReLU(inplace=True)
        # Block 6: 16 -> 3 channels
        # 建立转置卷积层：通道 16→3；它通过可训练参数上采样，不是把编码运算精确倒过来。
        self.deconv6 = nn.ConvTranspose2d(16, 3, (2, 2), stride=(2, 2))  # 输出RGB图像
        # 建立 ReLU 激活层：保留正值、把负值变成 0；inplace=True 表示尽可能原地修改张量。
        self.relu6 = nn.ReLU(inplace=True)
    # 定义模型前向计算；外部写 model(x) 时，PyTorch 的调用机制会执行这里的运算。
    # x 是张量，后面的 x = ... 表示把中间结果继续传给下一层，不是反向传播。
    def forward(self, x):
        """前向传播：通过转置卷积逐步上采样"""
        # 输入为 [batch,512]，下面补齐两个空间维度
        # 在最后加两个长度为 1 的轴：[B,512]→[B,512,1,1]，才能交给二维转置卷积。
        x = x.unsqueeze(-1).unsqueeze(-1)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv1(x)  # -> (batch, 256, 2, 2)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu1(x)
        # 输入为 (batch, 256, 2, 2)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv2(x)  # -> (batch, 128, 4, 4)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu2(x)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv3(x)  # -> (batch, 64, 8, 8)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu3(x)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv4(x)  # -> (batch, 32, 16, 16)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu4(x)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv5(x)  # -> (batch, 16, 32, 32)
        # 执行 ReLU 激活：把负值设为 0，保留非负特征。
        x = self.relu5(x)
        # 当前转置卷积恢复一层空间分辨率，高宽各扩大两倍，同时改变通道数。
        x = self.deconv6(x)  # -> (batch, 3, 64, 64)
        # 最后用 sigmoid 限定重建像素范围为 (0,1)，与训练目标的 [0,1] 范围配合。
        x = x.sigmoid()  # 与 [0,1] 训练目标匹配
        # 将本网络/当前阶段的最终张量返回给调用方；本函数到这里结束。
        return x  # 输出重建后的图像

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 导入 torch，使用张量、神经网络计算及自动求导
    import torch
    # 生成 [0,1) 的随机浮点张量，作为测试输入；它不是来自真实图片的像素。
    encoded = ConvEncoder()(torch.rand(1, 3, 64, 64))
    # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
    print(encoded.shape, ConvDecoder()(encoded).shape)
