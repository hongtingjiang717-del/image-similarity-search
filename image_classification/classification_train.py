# 【初学者阅读说明】
# 分类训练入口；输入图片，学习预测五类商品中的一个。
# 阅读顺序与去噪入口相同，区别是数据目标为类别整数、损失为交叉熵、模型为 Classifier。
# 运行：python -m image_classification.classification_train --epochs 1 --limit 16 --batch-size 4

"""运行：python -m image_classification.classification_train --device cpu。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.training 导入 training_parser, make_loaders, fit；函数/类在对应文件实现，本文件直接复用。
from common.training import training_parser, make_loaders, fit
# 从 common.utils 导入 seed_everything, get_device；函数/类在对应文件实现，本文件直接复用。
from common.utils import seed_everything, get_device
# 包内相对导入：前面的点表示当前包；从 本包 取出 classification_config as config，避免误找同名顶层模块。
from . import classification_config as config
# 包内相对导入：前面的点表示当前包；从 classification_data 取出 ImageDataset，避免误找同名顶层模块。
from .classification_data import ImageDataset
# 包内相对导入：前面的点表示当前包；从 classification_model 取出 Classifier，避免误找同名顶层模块。
from .classification_model import Classifier
# 包内相对导入：前面的点表示当前包；从 classification_engine 取出 train_step, val_step，避免误找同名顶层模块。
from .classification_engine import train_step, val_step

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 建立本任务的命令行解析器，并注册数据目录、轮数、批次大小等通用选项。
    parser = training_parser(__doc__, config)
    # 登记命令行参数 --labels；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--labels", type=Path, default=config.FASHION_LABELS_PATH)
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 在随机初始化和划分数据之前固定随机源，提高同一环境中实验的可重复性。
    seed_everything(args.seed)
    # 读取或选择当前计算设备，之后网络与输入都放到此处进行运算。
    device = get_device(args.device)
    # 创建统一图像预处理：缩放尺寸、转成 [C,H,W] 浮点张量。
    dataset = ImageDataset(args.data_dir, args.labels, build_transform(config.IMG_HEIGHT, config.IMG_WIDTH))
    # 按固定随机种子划分数据，并返回训练、验证和全量三个批次加载器。
    train, val, _ = make_loaders(dataset, args, config.TRAIN_RATIO)
    # 实例化所需网络并移动到计算设备；此时参数仍需训练，或随后从权重文件加载。
    model = Classifier().to(device)
    # 创建 AdamW 优化器，登记要学习的参数与学习率；只有调用 step 时才更新参数。
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # 开始完整训练：把模型、损失、优化器、训练/验证数据传给公共 fit，由它按轮调用引擎并保存最优参数。
    fit([model], [config.CLASSIFIER_MODEL_NAME], train_step, val_step, train, val,
        torch.nn.CrossEntropyLoss(), optimizer, args, device)

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
