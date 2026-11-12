# 【初学者阅读说明】
# 这是参数真正发生学习的地方：一次调用遍历一轮（epoch）的全部批次。
# 训练：清除旧梯度 → 前向计算 → 算损失 → backward 求梯度 → step 更新参数。
# 验证：前向计算并统计损失，不计算梯度、不更新参数。optimizer=None 用来区分验证。
# models 是模型对象列表；forward 是可调用对象；loader 按批给出输入和目标。

"""统一处理训练模式、梯度、按样本加权的损失和空数据检查。"""
# 从 contextlib 导入 nullcontext，用于提供上下文管理器，例如验证时切换梯度记录方式或测试时捕获打印。
from contextlib import nullcontext
# 导入 torch，使用张量、神经网络计算及自动求导
import torch

# optimizer 有值时训练，值为 None 时只验证；返回本轮按样本加权的平均损失。
# forward 不限定具体网络：可以是单个模型，也可以是“先编码再解码”的 lambda。
def run_epoch(models, forward, loader, loss_fn, device, optimizer=None):
    # is not None 判断是否真的传入优化器；这里得到布尔值 True（训练）或 False（验证）。
    training = optimizer is not None
    # for 循环逐项遍历 models，每轮把当前元素赋给 model，再执行缩进内操作。
    for model in models:
        # 让模型切到训练/评估模式；train(False) 等同 eval()，它本身不会更新参数。
        model.train(training)
    # 同时初始化损失总和与样本总数；逗号左右一一对应，这是多目标赋值。
    total_loss, total_samples = 0.0, 0
    # 条件表达式：训练时保持正常求导，验证时用 no_grad 关闭梯度记录。
    context = nullcontext() if training else torch.no_grad()
    # 进入刚才选择的上下文：训练保留梯度，验证关闭梯度；退出时自动恢复状态。
    with context:
        # 每次从加载器取一批（输入，目标）并解包；一轮循环会处理完所有批次。
        for inputs, targets in loader:
            # 把一批输入和目标都移到相同计算设备；模型与张量必须在匹配的设备上运算。
            inputs, targets = inputs.to(device), targets.to(device)
            # 只有训练阶段才执行下面的清梯度或参数更新；验证阶段跳过。
            if training:
                # 清除上一批的梯度，防止默认的梯度累加干扰本次更新；set_to_none 用 None 表示尚未计算。
                optimizer.zero_grad(set_to_none=True)
            # 调用传入的前向函数得到预测；这里的函数可代表一个模型，也可代表编码器接解码器。
            output = forward(inputs)
            # 检查输入/输出的维度与大小是否符合当前网络约定；不匹配时提前提示，而不是让矩阵运算难懂地报错。
            if output.shape != targets.shape and targets.ndim != 1:
                # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                raise ValueError(f"输出和目标尺寸不一致：{output.shape} / {targets.shape}")
            # 用损失函数比较预测和目标，得到供反向传播使用的单个损失张量。
            loss = loss_fn(output, targets)
            # 检查损失是否为 NaN 或无穷大；异常数值应立即停止，不能继续当有效训练。
            if not torch.isfinite(loss):
                # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                raise RuntimeError("损失出现 NaN 或无穷大，请检查数据和学习率。")
            # 只有训练阶段才执行下面的清梯度或参数更新；验证阶段跳过。
            if training:
                # 从损失出发，按链式法则计算各可训练参数的梯度；梯度表示损失对参数的变化率。
                loss.backward()
                # 优化器读取参数的梯度并调整参数；这是模型真正“学习、改变权重”的一步。
                optimizer.step()
            # 第 0 维是本批图片张数；尾批可能不足 batch_size，所以读取实际张数。
            count = inputs.size(0)
            # item 取出单元素损失的 Python 数值；乘实际样本数再累加，避免小尾批权重过大。
            total_loss += loss.item() * count
            # 把本批实际张数加到累计样本数；+= 等价于先相加再赋回原变量。
            total_samples += count
    # 防止加载器没有样本时除以 0，也明确提示训练/验证数据为空。
    if total_samples == 0:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("数据加载器为空，无法计算平均损失。")
    # 用总的加权损失除以样本总数，得到一轮中每张图片的平均损失。
    return total_loss / total_samples
