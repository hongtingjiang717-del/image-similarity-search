#==================定义模型======================
import torch.nn as nn
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray


# 由于相似检索模块和去噪模块虽然都是自编码器，但是相似检索模块需要需要自编码器中编码的输出向量加入到chroma库中
# 所以自编码器的编码器和解码器部分我们分开来写
# 分别自编码器模型中的自编码器和解码器
#编码器部分
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        ## 编码器层 ##
        # 卷积层 (输入通道数从1变为32), 3x3卷积核
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层 (输入通道数从32变为16), 3x3卷积核
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)

        # 池化层，用于将x-y维度减少一半；卷积核和步幅均为2
        self.pool = nn.MaxPool2d(2, 2) #见一面砍一半

    # forward函数用于向前传播，该函数接受一个输入张量x，并返回一个输出张量，该函数的实现真正决定了神经网络的架构
    #前向传播-编码，得到长度为512的向量
    def forward(self, x):
        ## 编码 ##
        # 添加带有ReLU激活函数的隐藏层
        # 并在之后进行最大池化
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        # 添加第二个隐藏层
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        # 添加第三个隐藏层
        x = torch.relu(self.conv3(x))
        x = self.pool(x)  # 压缩表示
        # 添加第四个隐藏层
        x = torch.relu(self.conv4(x))
        x = self.pool(x)  # 压缩表示
        # 添加第五个隐藏层
        x = torch.relu(self.conv5(x))
        x = self.pool(x)  # 压缩表示
        # 添加第六个隐藏层
        x = torch.relu(self.conv6(x))
        x = self.pool(x)  # 压缩表示

        #最后的输出维度是[N,512,1,1]
        #将[N,512,1,1]的张量做向量化处理，得到[N,512]
        x = x.squeeze(-1).squeeze(-1)
        return x

#解码器部分
class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        ## 解码器层
        # 转置卷积层，卷积核为2，步幅为2，将空间维度增加2倍
        self.t_conv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv6 = nn.ConvTranspose2d(16, 3, 2, stride=2)


        # forward函数用于向前传播，该函数接受一个输入张量x，并返回一个输出张量，该函数的实现真正决定了神经网络的架构
    #前向传播-解码，恢复原始图像
    def forward(self, x):
        #先将（N,512）的一组向量恢复为（N,512,1,1）
        x = x.unsqueeze(-1).unsqueeze(-1)
        ## 解码
        # 添加转置卷积层，带有ReLU激活函数
        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        x = torch.relu(self.t_conv3(x))
        x = torch.relu(self.t_conv4(x))
        x = torch.relu(self.t_conv5(x))
        # 再次转置卷积，输出应应用sigmoid函数
        x = torch.sigmoid(self.t_conv6(x))
        return x
#测试
import torch

if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor = torch.randn(10, 3, 64, 64)
    encoder = ConvEncoder()
    decoder =  ConvDecoder()
    #编码
    vectors = encoder(input_tensor)
    #解码
    output = decoder(vectors)
    print(vectors.shape)
    print(output.shape)
