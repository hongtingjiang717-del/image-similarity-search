# 【初学者阅读说明】
# 本文件提供几个可重复使用的小工具，训练和推理都会调用它。
# seed_everything 固定随机源；get_device 选择 CPU/GPU；save_weights/load_weights 保存/恢复参数。
# 模型对象好比机器结构，权重好比机器调好的旋钮；只有结构或只有旋钮都不能完成正确推理。

"""随机种子、设备选择和权重读写。"""
# 导入 random，控制 Python 自带的随机数发生器
import random
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 导入 numpy，处理数值数组、保存或读取 npy 向量文件；as np 给它起较短别名
import numpy as np
# 导入 torch，使用张量、神经网络计算及自动求导
import torch

# 同一种子使随机序列便于重复；固定种子不意味着每次抽出的随机数都相同。
def seed_everything(seed):
    """固定当前进程中常用随机源；不承诺跨硬件、跨版本完全一致。"""
    # 固定 Python 自带随机模块的随机种子。
    random.seed(seed)
    # 固定 NumPy 旧式全局随机数接口的种子。
    np.random.seed(seed)
    # 设置 PyTorch 的随机种子，影响随机初始化和此后相关随机抽样。
    torch.manual_seed(seed)
    # 条件分支：只有 torch.cuda.is_available() 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if torch.cuda.is_available():
        # 仅在 CUDA 可用的分支中，为所有可见 GPU 设置随机种子。
        torch.cuda.manual_seed_all(seed)
    # 要求相关 cuDNN 算法优先采用确定性实现；不同硬件/版本仍不保证完全一致。
    torch.backends.cudnn.deterministic = True
    # 关闭按当前输入测速选择卷积算法的机制，减少复现实验中的算法选择变化。
    torch.backends.cudnn.benchmark = False

# name 可以是 cpu、auto、cuda 或 cuda:0；返回 torch.device 对象。
def get_device(name="cpu"):
    # 条件分支：只有 name == 'auto' 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if name == "auto":
        # 只有用户选择 auto 时才按可用性选设备；三元表达式在 GPU 可用时取 cuda，否则取 cpu。
        name = "cuda" if torch.cuda.is_available() else "cpu"
    # 把设备字符串转成 PyTorch 设备对象；CPU 与 GPU 上的模型和张量要匹配。
    device = torch.device(name)
    # 条件分支：只有 device.type == 'cuda' and (not torch.cuda.is_available()) 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if device.type == "cuda" and not torch.cuda.is_available():
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise RuntimeError("当前 PyTorch 无可用 CUDA，请使用 --device cpu。")
    # 把 device 交还给调用方，并结束本次函数调用；return 不会打印结果。
    return device

# 输入 model（网络对象）与 path（保存路径）；保存 state_dict，不保存整个类定义。
def save_weights(model, path):
    """先写临时文件再替换，减少中断时留下损坏权重的风险。"""
    # 把字符串或已有路径统一包装成 Path 对象，以便调用 mkdir、is_file 等方法。
    path = Path(path)
    # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 生成旁边的临时文件名，例如 denoiser.pt.tmp；不直接写坏已有完整文件。
    temporary = path.with_suffix(path.suffix + ".tmp")
    # 把模型参数字典写入临时文件；state_dict 的键是层名，值是参数或持久状态张量。
    torch.save(model.state_dict(), temporary)
    # 临时文件写完后替换目标文件；尽量缩小写入中断导致损坏的窗口。
    temporary.replace(path)

# 先创建相同结构的模型，再读入参数；返回已移动设备并切换评估模式的模型。
def load_weights(model, path, device="cpu"):
    # 把字符串或已有路径统一包装成 Path 对象，以便调用 mkdir、is_file 等方法。
    path = Path(path)
    # 先检查必要的目录/文件是否存在；不存在时给出可理解的提示。
    if not path.is_file():
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise FileNotFoundError(f"未找到模型权重：{path}。请先训练对应模型，或指定正确的权重路径。")
    # 把参数先读到 CPU，按 weights_only 模式加载；不是读取完整 Python 模型类。
    state = torch.load(path, map_location="cpu", weights_only=True)
    # 把保存的参数填入同结构模型；层名或形状不匹配会报错，而不是静默使用错误结构。
    model.load_state_dict(state)
    # 链式调用：先移动模型参数到设备，再切换评估模式，最后返回这个模型对象。
    return model.to(device).eval()
