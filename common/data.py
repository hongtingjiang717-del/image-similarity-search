# 本文件把“硬盘上的图片”转换成“模型能计算的数字数组（张量）”。
# 读取顺序：list_images 收集路径 → load_image 打开图片 → transform 缩放并转成张量。
# 普通 RGB 图片转为 [C,H,W]=[3,64,64]；DataLoader 组批后成为 [B,3,64,64]。
# B 是这批图片张数；C 是颜色通道；H 是高度；W 是宽度；这些字母只是形状记号。

"""三个模块共享的图片枚举和预处理；不自动跳过损坏图片。"""
# 导入 re，用正则表达式拆分文件名中的数字和文字
import re
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 从 PIL 导入 Image，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image
# 从 torch.utils.data 导入 Dataset，用于使用张量、神经网络计算及自动求导。
from torch.utils.data import Dataset
# 导入 torchvision.transforms，使用图像缩放与张量转换等视觉工具；as T 给它起较短别名
import torchvision.transforms as T

# 集合 set：列出允许读取的图片后缀，in 可检查某后缀是否受支持。
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 输入一组文件名或路径；返回按自然数字顺序排列的新列表。
def sorted_alphanumeric(data):
    # 把 img12.jpg 拆成文字和整数，使 img2.jpg 排在 img12.jpg 前面。
    def natural_key(value):
        # 用列表推导式整理输出，保持数据库返回的顺序；每一项是一个结果记录或文件名。
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"([0-9]+)", str(value))]
    # sorted 返回新列表；key= 指定“按什么值比较”，让文件名数字按数值而非字符顺序排序。
    return sorted(data, key=natural_key)

# 只枚举当前目录中的受支持图片文件，不递归子目录；返回排序后的 Path 列表。
def list_images(main_dir):
    # 把图片目录转换为 Path 对象，后续可检查是否存在并枚举文件。
    root = Path(main_dir)
    # 先检查必要的目录/文件是否存在；不存在时给出可理解的提示。
    if not root.is_dir():
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise FileNotFoundError(f"图片目录不存在：{root}")
    # 用列表推导式过滤出支持后缀的普通图片文件，再按文件名自然排序；目录不当图片读。
    paths = sorted_alphanumeric([p for p in root.iterdir()
                                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    # 条件分支：只有 not paths 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if not paths:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError(f"图片目录没有支持的图片：{root}")
    # 把 paths 交还给调用方，并结束本次函数调用；return 不会打印结果。
    return paths

# 返回可调用的预处理流水线：统一尺寸，再把 PIL 图片转为 [C,H,W] 浮点张量。
def build_transform(height=64, width=64):
    # Compose 按列表顺序执行：Resize 统一尺寸，ToTensor 把普通 RGB 像素从 0～255 转为 [0,1] 的 [3,H,W] 张量。
    return T.Compose([T.Resize((height, width)), T.ToTensor()])

# 定义 ImageFolderDataset 数据集类并继承父类能力；只重写与当前任务不同的部分。
class ImageFolderDataset(Dataset):
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self, main_dir, transform=None):
        # 保存图片主目录；self.main_dir 属于当前数据集对象，不是所有数据集共享的全局变量。
        self.main_dir = Path(main_dir)
        # 如果调用者给了预处理就使用它，否则构建默认 64×64 预处理；这是三元条件表达式。
        self.transform = transform if transform is not None else build_transform()
        # 收集支持的真实图片路径，并固定自然排序；后续图片、标签、向量都依赖这个顺序。
        self.paths = list_images(main_dir)
        # 列表推导式：遍历每条路径 p，只取文件名 name，组成新列表。
        self.all_imgs = [p.name for p in self.paths]

    # Python 执行 len(数据集对象) 时自动调用这个方法，得到图片数量。
    def __len__(self):
        # 返回图片列表长度；len(dataset) 会间接得到这里的值。
        return len(self.paths)

    # idx 是图片在名册中的下标，从 0 开始；返回预处理后的单张图片张量。
    def load_image(self, idx):
        # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
        with Image.open(self.paths[idx]) as image:
            # 先统一为 RGB 三通道，再调用预处理；返回的张量已经不依赖打开的图片文件句柄。
            return self.transform(image.convert("RGB"))

    # Python 执行 dataset[idx] 时自动调用这个方法；DataLoader 也通过它逐个取样本。
    def __getitem__(self, idx):
        # 按索引加载并预处理一张图片，得到 [3,H,W] 数字张量。
        image = self.load_image(idx)
        # 返回二元组（输入，目标）；自编码器用原图作为输入，也用原图作为重建目标。
        return image, image
