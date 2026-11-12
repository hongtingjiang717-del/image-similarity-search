# 【初学者阅读说明】
# 训练/验证时，图片先进入 encoder，再进入 decoder，最后比较重建图和原图。
# create_embedding 只调用编码器，按批收集 [B,512]，最后拼成 [N,512]。
# 这里导出的 npy 是原始向量；Chroma 建库时会另做 L2 归一化。
# 此文件是任务接口；真正共享的循环实现在 common/engine.py。
# train_step 有 optimizer 参数会更新权重，val_step 没有优化器只统计效果。

"""一个 epoch 的训练和验证；共享引擎负责模式切换与损失汇总。"""
# 从 common.engine 导入 run_epoch；函数/类在对应文件实现，本文件直接复用。
from common.engine import run_epoch

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["train_step", "val_step"]

# 遍历一轮训练数据，并更新参数；device 是计算设备，loss_fn 衡量预测与目标的差异。
def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    # 把当前模型及前向计算方式交给公共引擎，直接返回一整轮的平均损失。
    return run_epoch([encoder, decoder], lambda x: decoder(encoder(x)), train_loader, loss_fn, device, optimizer)

# 遍历一轮验证数据，只衡量当前模型效果；不接收优化器，不更新参数。
def val_step(encoder, decoder, val_loader, loss_fn, device):
    # 把当前模型及前向计算方式交给公共引擎，直接返回一整轮的平均损失。
    return run_epoch([encoder, decoder], lambda x: decoder(encoder(x)), val_loader, loss_fn, device)


# 输入编码器与全量加载器，返回 CPU 上的 [N,512] 张量，并可另存为 npy 文件。
def create_embedding(encoder, full_loader, embedding_name=None, device="cpu"):
    """分批生成 [N,512] 特征，保存 npy；不再调用不存在的函数。"""
    # 导入 numpy，处理数值数组、保存或读取 npy 向量文件；as np 给它起较短别名
    import numpy as np
    # 导入 torch，使用张量、神经网络计算及自动求导
    import torch
    # 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
    from pathlib import Path
    # 切换评估模式；对 Dropout/BatchNorm 等层有效，但不会自动关闭梯度计算。
    encoder.eval()
    # 建立批次结果列表；每次编码一批就把该批 CPU 特征加入列表。
    batches = []
    # with torch.no_grad() 临时关闭梯度记录，退出块后恢复原状态；它不等同于 model.eval()。
    with torch.no_grad():
        # 每次从加载器取一批（输入，目标）并解包；一轮循环会处理完所有批次。
        for images, _ in full_loader:
            # 每批图移到编码器设备后提取并展平特征，结果移回 CPU 加入列表；不保存整批原图片。
            batches.append(encoder(images.to(device)).flatten(1).cpu())
    # 条件分支：只有 not batches 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if not batches:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("数据加载器为空，无法导出嵌入")
    # 沿批次轴连接所有批特征，例如 [3,512]+[2,512]→[5,512]，保留样本顺序。
    embedding = torch.cat(batches, dim=0)
    # 条件分支：只有 embedding_name is not None 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if embedding_name is not None:
        # 把特征文件名转换为路径对象，供创建目录与 npy 保存使用。
        path = Path(embedding_name)
        # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保存二维浮点数组为 npy 文件；此文件没有文件名，需与另存的映射配套使用。
        np.save(path, embedding.numpy())
    # 返回 CPU 特征张量 [N,512]，不是重建后的图片。
    return embedding

# 把特征导出函数名补充到模块公开接口列表，星号导入时也能找到它。
__all__.append("create_embedding")
