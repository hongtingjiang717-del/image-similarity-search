# 【初学者阅读说明】
# 此文件是任务接口；真正共享的循环实现在 common/engine.py。
# train_step 有 optimizer 参数会更新权重，val_step 没有优化器只统计效果。

"""一个 epoch 的训练和验证；共享引擎负责模式切换与损失汇总。"""
# 从 common.engine 导入 run_epoch；函数/类在对应文件实现，本文件直接复用。
from common.engine import run_epoch

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["train_step", "val_step"]

# 遍历一轮训练数据，并更新参数；device 是计算设备，loss_fn 衡量预测与目标的差异。
def train_step(denoiser, train_loader, loss_fn, optimizer, device):
    # 把当前模型及前向计算方式交给公共引擎，直接返回一整轮的平均损失。
    return run_epoch([denoiser], denoiser, train_loader, loss_fn, device, optimizer)

# 遍历一轮验证数据，只衡量当前模型效果；不接收优化器，不更新参数。
def val_step(denoiser, val_loader, loss_fn, device):
    # 把当前模型及前向计算方式交给公共引擎，直接返回一整轮的平均损失。
    return run_epoch([denoiser], denoiser, val_loader, loss_fn, device)
