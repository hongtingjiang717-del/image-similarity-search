# python项目结构
image_similarity_main/
│
├── common/                 # 公共模块
│   ├── __init__.py
│   ├── utils.py            # 工具函数
│   ├── dataset/            # 商品图片
│   └── fashion-labels.csv  # 类别标签
│
├── image_denoising/        # 图像去噪
│   ├── denoising_config.py
│   ├── denoising_data.py
│   ├── denoising_model.py
│   ├── denoising_engine.py
│   └── denoising_train.py
│
├── image_classification/   # 商品分类
│   ├── classification_config.py
│   ├── classification_data.py
│   ├── classification_model.py
│   ├── classification_engine.py
│   └── classification_train.py
│
├── image_similarity/       # 相似图像检索
│   ├── similarity_config.py
│   ├── similarity_data.py
│   ├── similarity_model.py
│   ├── similarity_engine.py
│   ├── similarity_train.py
│   └── similarity_embeddings.py
│
└── web/                    # Web 前后端
    ├── templates/
    └── web_app.py


# python的基础知识
概念	通俗理解	本项目中的例子
模块 Module	一个包含 Python 代码的文件	utils.py
包 Package	组织多个模块的目录	common/
库 Library	可复用的代码集合	NumPy、PyTorch
类 Class	创建对象的蓝图	ImageDataset
对象 Object	根据类创建的实例	model = ConvDenoiser()
函数 Function	执行特定任务的一段代码	seed_everything()
方法 Method	定义在类中的函数	model.forward()

# conda环境的准备
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
经验：在安装库的时候，加上一步换源操作就可以直接将下边需要安装的所有库从国外的软件仓库下载改为从国外的镜像站下载，加快python第三方的下载速度，提高安装成功率。

# 去噪模块
1.自编码器：包括编码器（卷积和池化）和解码器（转置卷积---反卷积，是上采样的一种方式）
转置卷积只能恢复数据的空间尺寸，却不能保证恢复原始的数值，是因为普通卷积（编码器中）是一个多对一的过程，本身就会丢失信息；stide、池化、非线性激活也会损失信息。
https://chatgpt.com/s/t_6aeadb709a4c819186964b87877ea186

## 为什么编码器中在卷积的时候，通道数一直要变化（往往是变多）？而图像的尺寸反而不变化？
输出通道数  等于  卷积核的个数
卷积层经常通过增加通道数来提升特征表达能力，同时通过 stride 和 padding 控制是否改变图像尺寸。
H_out = floor((H_in + 2*padding - kernel_size) / stride) + 1
当 stride=1 且 padding 刚好抵消卷积核带来的缩小，比如 kernel_size=3, padding=1，那么输出尺寸就会和输入尺寸相同。57
所以“尺寸不变”不是卷积的必然结果，而是人为设计出来的：
想保留空间分辨率，就用 stride=1 + 合适 padding；
想缩小尺寸，就用 stride>1 或池化层；
想只改通道不改尺寸，就用 1×1 卷积。

## dataset和dataloder有什么区别？
Dataset = 数据仓库 / 菜谱：负责“一个样本怎么拿”。
DataLoader = 传送带 / 服务员：负责“一批一批怎么送”。

## __getitem__根据索引去拿取图像