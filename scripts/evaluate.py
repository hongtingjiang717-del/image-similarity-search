# 【初学者阅读说明】
# 本脚本加载训练保存的最佳权重，并按 split.json 指定的验证文件名进行评估。
# 分类：平均交叉熵和准确率；去噪/重建：平均 MSE。不会更新模型参数。
# 准确率=预测正确张数/验证总张数；重建损失不能直接代表相似检索的相关性。

"""使用训练时保留的验证集评估最优权重；不使用检索库报告训练效果。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 json，把字典/列表与 JSON 文本相互转换
import json
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 torch.utils.data 导入 DataLoader, Subset，用于使用张量、神经网络计算及自动求导。
from torch.utils.data import DataLoader, Subset
# 从 common.paths 导入 ARTIFACT_DIR, DATA_DIR, LABELS_PATH, DEFAULT_DEVICE；函数/类在对应文件实现，本文件直接复用。
from common.paths import ARTIFACT_DIR, DATA_DIR, LABELS_PATH, DEFAULT_DEVICE
# 从 common.utils 导入 get_device, load_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import get_device, load_weights


# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记命令行参数 task；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 choices 限制允许输入的取值。
    parser.add_argument("task", choices=["denoising", "classification", "similarity"])
    # 登记命令行参数 --artifact-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    # 登记命令行参数 --data-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    # 登记命令行参数 --labels；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    # 登记命令行参数 --device；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 按所选任务名定位该模型的训练产物目录。
    folder = args.artifact_dir / args.task
    # 把 JSON 文件文本解析成 Python 字典/列表，恢复训练参数或文件名清单。
    metadata = json.loads((folder / "training_config.json").read_text(encoding="utf-8"))
    # 读取或选择当前计算设备，之后网络与输入都放到此处进行运算。
    device = get_device(args.device)
    # 根据任务选择分类专用模型、标签和统计方式。
    if args.task == "classification":
        # 从 image_classification.classification_data 导入 ImageDataset；函数/类在对应文件实现，本文件直接复用。
        from image_classification.classification_data import ImageDataset
        # 从 image_classification.classification_model 导入 Classifier；函数/类在对应文件实现，本文件直接复用。
        from image_classification.classification_model import Classifier
        # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
        dataset = ImageDataset(args.data_dir, args.labels)
        # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
        model = load_weights(Classifier(), folder / "classifier.pt", device)
        # 创建交叉熵损失：输入五类原始分数与正确类别整数，不需要先 softmax。
        loss_fn = torch.nn.CrossEntropyLoss()
    # 条件分支：只有 args.task == 'denoising' 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    elif args.task == "denoising":
        # 从 image_denoising.denoising_data 导入 ImageDataset；函数/类在对应文件实现，本文件直接复用。
        from image_denoising.denoising_data import ImageDataset
        # 从 image_denoising.denoising_model 导入 ConvDenoiser；函数/类在对应文件实现，本文件直接复用。
        from image_denoising.denoising_model import ConvDenoiser
        # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
        dataset = ImageDataset(args.data_dir, noise_seed=metadata["seed"])
        # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
        model = load_weights(ConvDenoiser(), folder / "denoiser.pt", device)
        # 创建均方误差损失：比较重建像素与目标像素的平方差并求平均。
        loss_fn = torch.nn.MSELoss()
    # else：前面的对应条件不成立（或 try 无异常）时，执行这个分支。
    else:
        # 从 image_similarity.similarity_data 导入 ImageDataset；函数/类在对应文件实现，本文件直接复用。
        from image_similarity.similarity_data import ImageDataset
        # 从 image_similarity.similarity_model 导入 ConvEncoder, ConvDecoder；函数/类在对应文件实现，本文件直接复用。
        from image_similarity.similarity_model import ConvEncoder, ConvDecoder
        # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
        dataset = ImageDataset(args.data_dir)
        # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
        model = torch.nn.Sequential(
            load_weights(ConvEncoder(), folder / "deep_encoder.pt", device),
            load_weights(ConvDecoder(), folder / "deep_decoder.pt", device))
        # 创建均方误差损失：比较重建像素与目标像素的平方差并求平均。
        loss_fn = torch.nn.MSELoss()
    # 按训练时保存的真实文件名评估，不因后续增加图片而重新划分验证集。
    # 把 JSON 文件文本解析成 Python 字典/列表，恢复训练参数或文件名清单。
    split = json.loads((folder / "split.json").read_text(encoding="utf-8"))
    # 建立“文件名→数据集索引”的字典；enumerate 同时给出序号与路径。
    indices = {path.name: index for index, path in enumerate(dataset.paths)}
    # 集合差集：从应有的验证文件名中减去现有文件名，找出缺失图片。
    missing = set(split["validation"]) - indices.keys()
    # 非空集合/列表在 if 中为真；发现缺失数据时停止并给出说明。
    if missing:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError(f"验证集图片缺失：{sorted(missing)[:5]}")
    # 创建按批取样器；batch_size 控制每批张数，shuffle 决定是否打乱，num_workers 决定读图子进程数。
    validation = DataLoader(Subset(dataset, [indices[name] for name in split["validation"]]),
                            batch_size=metadata["batch_size"], shuffle=False)
    # 条件分支：只有 len(validation.dataset) == 0 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if len(validation.dataset) == 0:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("保存的验证集为空")
    # 分别累计验证样本数、损失加权和、分类预测正确数。
    total, loss_sum, correct = 0, 0.0, 0
    # 切换评估模式；对 Dropout/BatchNorm 等层有效，但不会自动关闭梯度计算。
    model.eval()
    # with torch.no_grad() 临时关闭梯度记录，退出块后恢复原状态；它不等同于 model.eval()。
    with torch.no_grad():
        # 每次从加载器取一批（输入，目标）并解包；一轮循环会处理完所有批次。
        for images, targets in validation:
            # 建立这一组初始数据；列表用 []、元组常用 ()、集合用 {}，逗号分隔各项。
            images, targets = images.to(device), targets.to(device)
            # 把这一批图片交给网络做前向运算，得到类别分数或重建图片。
            output = model(images)
            # 按本批实际图片数加权累计损失，稍后再除以验证总数。
            loss_sum += loss_fn(output, targets).item() * len(images)
            # 把右侧的本批统计值累加到左侧变量；统计完所有批次后再计算平均值。
            total += len(images)
            # 根据任务选择分类专用模型、标签和统计方式。
            if args.task == "classification":
                # 每张图取分数最高的类别，与正确标签逐个比较，把 True 当 1 求和，累计预测正确数。
                correct += int((output.argmax(1) == targets).sum().item())
    # 把样本数、损失或检查统计值组织为结果字典，随后转成 JSON 方便阅读。
    result = dict(task=args.task, validation_samples=total, loss=loss_sum / total)
    # 根据任务选择分类专用模型、标签和统计方式。
    if args.task == "classification":
        # 计算准确率，例如 80/100=0.8，即 80%；不能用它描述去噪或检索质量。
        result["accuracy"] = correct / total
    # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
    print(json.dumps(result, ensure_ascii=False, indent=2))


# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
