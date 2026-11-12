# 【初学者阅读说明】
# 本文件负责训练外围流程，具体卷积网络仍定义在各模块的 model 文件。
# training_parser 读命令行；make_loaders 分训练/验证集；fit 按轮训练并保存最优参数。
# args 是 argparse 生成的参数对象，例如 --epochs 1 会成为 args.epochs == 1。
# dataset 是整本图片名册；Subset 像名册的一份索引清单；DataLoader 再把图片一批批取出。
# 示例：8 张图按 75% 分成训练 6 张、验证 2 张；batch_size=3 时训练分两批。

"""共享命令行参数、数据划分和最佳权重保存，具体模型仍位于各模块。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 json，把字典/列表与 JSON 文本相互转换
import json
# 导入 math，使用 isfinite 等数学判断，检查学习率是否是正常有限数
import math
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 torch.utils.data 导入 DataLoader, Subset，用于使用张量、神经网络计算及自动求导。
from torch.utils.data import DataLoader, Subset
# 从 common.paths 导入 DEFAULT_DEVICE；函数/类在对应文件实现，本文件直接复用。
from common.paths import DEFAULT_DEVICE
# 从 common.utils 导入 save_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import save_weights

# 命令行传进来的值通常是字符串；把它转成正整数，拒绝 0 和负数。
def positive_int(value):
    # int 把终端字符串转换成整数；例如 "4" 变为 4，不能转换的文字会报错。
    value = int(value)
    # 条件分支：只有 value < 1 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if value < 1:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise argparse.ArgumentTypeError("参数必须是正整数")
    # 把 value 交还给调用方，并结束本次函数调用；return 不会打印结果。
    return value

# 输入 description（帮助文字）、config（配置模块）；返回参数解析器，还没有开始训练。
# 每次 add_argument 都登记一个选项；调用 parse_args 后才得到具体参数值。
def training_parser(description, config):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description=description)
    # 登记命令行参数 --data-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--data-dir", type=Path, default=config.IMG_PATH)
    # 登记命令行参数 --output-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    # 登记命令行参数 --device；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu（默认）、auto、cuda 或 cuda:0")
    # 登记命令行参数 --epochs；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=positive_int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--epochs", type=positive_int, default=config.EPOCHS)
    # 登记命令行参数 --batch-size；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=positive_int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--batch-size", type=positive_int, default=config.TRAIN_BATCH_SIZE)
    # 登记命令行参数 --lr；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=float 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    # 登记命令行参数 --seed；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--seed", type=int, default=config.SEED)
    # 登记命令行参数 --num-workers；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--num-workers", type=int, default=0)
    # 登记命令行参数 --limit；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=positive_int 会把终端字符串转换成指定类型。
    parser.add_argument("--limit", type=positive_int, help="仅抽取指定数量图片做流程验证；正式训练不传")
    # 把 parser 交还给调用方，并结束本次函数调用；return 不会打印结果。
    return parser

# 输入完整数据集和命令行参数；返回训练、验证、全量三个 DataLoader。
# val_dataset 可与训练集用不同读图方式：去噪验证固定噪声，但图片索引仍与训练集对应。
def make_loaders(dataset, args, train_ratio=0.75, val_dataset=None):
    # 条件分支：只有 args.num_workers < 0 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if args.num_workers < 0:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("num_workers 不能为负数")
    # 检查数值是否正常且向量非零，避免余弦距离计算遇到无效特征。
    if not math.isfinite(args.lr) or args.lr <= 0:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("学习率必须是有限正数")
    # 建立独立随机发生器并固定种子，让数据划分不易受其他随机操作影响。
    generator = torch.Generator().manual_seed(args.seed)
    # 生成 0～N-1 的随机排列，再转为 Python 列表；打乱的是索引，图片仍在原目录。
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    # 条件分支：只有 args.limit is not None 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if args.limit is not None:
        # 切片 [:n] 保留前 n 个随机索引，用于小样本验证；不是先取文件名最小的 n 张。
        indices = indices[:args.limit]
    # 条件分支：只有 len(indices) < 2 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if len(indices) < 2:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("至少需要两张图片，才能划分训练集和验证集。")
    # 算训练张数，并限制在 1～总数减 1 之间，确保训练与验证至少各有一张。
    train_size = max(1, min(len(indices) - 1, int(len(indices) * train_ratio)))
    # 把共用参数放进字典；之后 **kwargs 会按参数名展开传给 DataLoader。
    kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    # 创建按批取样器；batch_size 控制每批张数，shuffle 决定是否打乱，num_workers 决定读图子进程数。
    train = DataLoader(Subset(dataset, indices[:train_size]), shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed), **kwargs)
    # 创建按批取样器；batch_size 控制每批张数，shuffle 决定是否打乱，num_workers 决定读图子进程数。
    validation = DataLoader(Subset(val_dataset if val_dataset is not None else dataset,
                                  indices[train_size:]), shuffle=False, **kwargs)
    # 创建按批取样器；batch_size 控制每批张数，shuffle 决定是否打乱，num_workers 决定读图子进程数。
    full = DataLoader(Subset(dataset, sorted(indices)), shuffle=False, **kwargs)
    # 依次返回训练、验证、全量三个加载器；调用方可用 a,b,c 解包，_ 表示有意不用某项。
    return train, validation, full

# 输入模型列表、权重文件名列表、训练/验证函数、两个加载器、损失、优化器和参数。
# *models 把模型列表拆成位置参数：一个模型传一个参数，编码器/解码器则传两个。
# 返回每轮的损失记录 history；同时写入最佳权重、split.json 和训练配置。
def fit(models, names, train_step, val_step, train_loader, val_loader, loss_fn,
        optimizer, args, device):
    # 取得本次实验的保存目录；不同实验可指定不同路径，以免覆盖结果。
    output_dir = Path(args.output_dir)
    # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
    output_dir.mkdir(parents=True, exist_ok=True)
    # 避免无意覆盖现有训练结果；重新实验请换 output-dir。
    # 列表推导式筛出已经存在的目标权重；存在时停止，以免无意覆盖上次实验。
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    # 非空列表在 if 中为真；发现已有权重时停止，保护旧实验。
    if existing:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise FileExistsError(f"输出目录已包含权重，请指定新的 --output-dir：{existing}")
    # 建立划分记录字典，准备保存每张图片属于训练集还是验证集。
    split = {}
    # for 循环逐项遍历 [('train', train_loader), ('validation', val_loader)]，每轮把当前元素赋给 (key, loader)，再执行缩进内操作。
    for key, loader in [("train", train_loader), ("validation", val_loader)]:
        # DataLoader.dataset 是它所读取的 Subset；Subset.dataset 才是原始图片数据集。
        subset = loader.dataset
        # 把子集索引换回真实文件名，存入 train 或 validation 列表，后续评估不用重新随机划分。
        split[key] = [subset.dataset.paths[index].name for index in subset.indices]
    # 将结果文本写入指定文件，encoding="utf-8" 保证中文正确；JSON 序列化让其他程序也能读取。
    (output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    # 最佳损失先设正无穷，历史记录先设空列表；第一轮正常验证结果就能成为最佳值。
    best_loss, history = float("inf"), []
    # 按轮循环；range 的结束值不包含在内，所以通常写 epochs+1 才能显示 1 到 epochs。
    for epoch in range(1, args.epochs + 1):
        # 把模型列表拆开传给训练函数，再传数据、损失和优化器；执行一整轮训练并拿回平均损失。
        train_loss = train_step(*models, train_loader, loss_fn, optimizer, device)
        # 使用本轮训练后的参数评估验证集；只计算效果，不把验证集用于更新权重。
        val_loss = val_step(*models, val_loader, loss_fn, device)
        # 验证损失越低越好；只在创造历史最低记录时保存最佳模型。
        if val_loss < best_loss:
            # 把当前更低的验证损失记录为历史最佳，供下一轮比较。
            best_loss = val_loss
            # zip 将多个序列按位置配对；for 同时解包每一组，任一序列结束就停止。
            for model, name in zip(models, names):
                # 给当前模型保存最佳参数；多个模型按 zip 的配对关系各自保存到对应文件。
                save_weights(model, output_dir / name)
        # 用一个字典记录当前轮号、训练损失和验证损失。
        row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        # 把当前轮记录追加到历史列表末尾，不覆盖之前的轮次。
        history.append(row)
        # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
        print(f"Epoch {epoch}/{args.epochs}: train={train_loss:.6f}, val={val_loss:.6f}", flush=True)
        # 将结果文本写入指定文件，encoding="utf-8" 保证中文正确；JSON 序列化让其他程序也能读取。
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    # 字典推导式遍历 args 属性；Path 先转字符串，因为 JSON 不能直接保存 Path 对象。
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    # 向元数据字典补充最佳损失、样本数量、软件版本、设备和权重格式，方便复核本次实验。
    metadata.update(best_val_loss=best_loss, train_samples=len(train_loader.dataset),
                    val_samples=len(val_loader.dataset), torch_version=torch.__version__,
                    device=str(device), image_size=[64, 64], checkpoint_format="state_dict")
    # 将结果文本写入指定文件，encoding="utf-8" 保证中文正确；JSON 序列化让其他程序也能读取。
    (output_dir / "training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # 返回所有轮次的记录列表，调用者可以据此分析或绘制损失曲线。
    return history
