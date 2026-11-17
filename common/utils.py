import numpy as np  # 数值计算库
import os  # 操作系统接口库
import torch  # PyTorch深度学习框架
import random  # 随机数生成库


def seed_everything(seed):
    """
    为了保证训练过程可复现，使用确定的随机数种子。对 torch，numpy 和 random 都使用相同的种子。

    参数:
    - seed: 随机数种子（整数）
    """
    random.seed(seed)  # 设置Python内置随机数种子
    os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python哈希种子
    np.random.seed(seed)  # 设置NumPy随机数种子
    torch.manual_seed(seed)  # 设置PyTorch CPU随机数种子
    torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN操作确定性

# 对字符串列表按字母-数字排序
# 正则表达式相关库
import re
#使文件名按照数字的顺序排序（而不是按照字典序列）
def sorted_alphanumeric(data):
    """按字母数字混合顺序对文件名进行排序（例如：img1, img2, ..., img10）"""
    # 定义转换函数：将数字部分转换为整数，非数字部分转换为小写
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    # 生成排序键：用正则分割字符串，分别处理数字和非数字部分
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    # 按生成的键排序
    return sorted(data, key=alphanum_key)