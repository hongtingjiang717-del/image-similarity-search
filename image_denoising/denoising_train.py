# 【初学者阅读说明】
# 这是去噪训练入口，建议先读本文件，再跟进它调用的公共函数。
# 顺序：读参数 → 固定随机种子 → 建数据集 → 划分批次 → 建模型/损失/优化器 → fit。
# 运行例子：python -m image_denoising.denoising_train --epochs 1 --limit 16 --batch-size 4
# test 这个函数只负责保存对比图；真正的验证损失由 val_step 统计。

"""运行：python -m image_denoising.denoising_train --device cpu。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.training 导入 training_parser, make_loaders, fit；函数/类在对应文件实现，本文件直接复用。
from common.training import training_parser, make_loaders, fit
# 从 common.utils 导入 seed_everything, get_device, load_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import seed_everything, get_device, load_weights
# 包内相对导入：前面的点表示当前包；从 本包 取出 denoising_config as config，避免误找同名顶层模块。
from . import denoising_config as config
# 包内相对导入：前面的点表示当前包；从 denoising_data 取出 ImageDataset，避免误找同名顶层模块。
from .denoising_data import ImageDataset
# 包内相对导入：前面的点表示当前包；从 denoising_model 取出 ConvDenoiser，避免误找同名顶层模块。
from .denoising_model import ConvDenoiser
# 包内相对导入：前面的点表示当前包；从 denoising_engine 取出 train_step, val_step，避免误找同名顶层模块。
from .denoising_engine import train_step, val_step

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 建立本任务的命令行解析器，并注册数据目录、轮数、批次大小等通用选项。
    parser = training_parser(__doc__, config)
    # 登记命令行参数 --preview；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 store_true 表示写出开关就为 True，不写为 False。
    parser.add_argument("--preview", action="store_true", help="训练结束后保存对比图，不弹出窗口")
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 在随机初始化和划分数据之前固定随机源，提高同一环境中实验的可重复性。
    seed_everything(args.seed)
    # 读取或选择当前计算设备，之后网络与输入都放到此处进行运算。
    device = get_device(args.device)
    # 创建统一图像预处理：缩放尺寸、转成 [C,H,W] 浮点张量。
    transform = build_transform(config.IMG_HEIGHT, config.IMG_WIDTH)
    # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
    dataset = ImageDataset(args.data_dir, transform)
    # 创建单独的验证读图对象，按同一图片索引固定噪声；划分时只取验证图片。
    validation = ImageDataset(args.data_dir, transform, noise_seed=args.seed)
    # 按固定随机种子划分数据，并返回训练、验证和全量三个批次加载器。
    train, val, _ = make_loaders(dataset, args, config.TRAIN_RATIO, validation)
    # 实例化所需网络并移动到计算设备；此时参数仍需训练，或随后从权重文件加载。
    model = ConvDenoiser().to(device)
    # 创建 Adam 优化器；parameters() 提供模型参数，lr 指学习率，不是训练轮数。
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 开始完整训练：把模型、损失、优化器、训练/验证数据传给公共 fit，由它按轮调用引擎并保存最优参数。
    fit([model], [config.DENOISER_MODEL_NAME], train_step, val_step, train, val,
        torch.nn.MSELoss(), optimizer, args, device)
    # 只有终端传入 --preview 开关时，才保存训练后的图片对比。
    if args.preview:
        # 加载验证最优的权重，再做预览或导出；避免误用最后一轮可能较差的模型。
        load_weights(model, args.output_dir / config.DENOISER_MODEL_NAME, device)
        # 使用加载后的最佳去噪模型保存验证图片对比图，不重新训练。
        test(model, val, device, args.output_dir / "preview.png")

# 取一批验证图，保存“带噪图/干净图/恢复图”三行对比，不弹出阻塞窗口。
def test(denoiser, val_loader, device, output_path="denoising_preview.png"):
    # 导入 matplotlib，把图片张量绘制成可查看的对比图
    import matplotlib
    # 使用 Agg 无窗口后端，只生成图片文件，便于 CPU 或无桌面的机器执行。
    matplotlib.use("Agg")
    # 导入 matplotlib.pyplot，把图片张量绘制成可查看的对比图；as plt 给它起较短别名
    import matplotlib.pyplot as plt
    # 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
    from pathlib import Path
    # iter 建立迭代器，next 只取第一批，解包为带噪图和干净图；不是遍历全部验证集。
    noisy, clean = next(iter(val_loader))
    # 切换评估模式；对 Dropout/BatchNorm 等层有效，但不会自动关闭梯度计算。
    denoiser.eval()
    # with torch.no_grad() 临时关闭梯度记录，退出块后恢复原状态；它不等同于 model.eval()。
    with torch.no_grad():
        # 将输入或中间张量移动到模型所在的计算设备；数值含义不因移动设备而改变。
        output = denoiser(noisy.to(device)).cpu()
    # 最多展示 10 张；当前批次不足 10 张时只画实际张数，避免越界。
    count = min(10, len(noisy))
    # 创建 3 行 count 列的子图；squeeze=False 确保只有一列时 axes 仍是二维数组。
    fig, axes = plt.subplots(3, count, squeeze=False, figsize=(count * 2, 6))
    # zip 将多个序列按位置配对；for 同时解包每一组，任一序列结束就停止。
    for images, row in zip([noisy, clean, output], axes):
        # zip 将多个序列按位置配对；for 同时解包每一组，任一序列结束就停止。
        for image, ax in zip(images[:count], row):
            # 把 [H,W,C] 数组绘制在子图中；不要直接把 [C,H,W] 的 PyTorch 排列交给它。
            ax.imshow(image.permute(1, 2, 0).numpy())
            # 设置绘图坐标轴显示；off 只隐藏坐标刻度，不改变图片像素。
            ax.axis("off")
    # 把预览图片的保存位置包装为 Path；接下来先创建父目录再保存。
    path = Path(output_path)
    # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 自动调整子图间距，尽量避免图像和边界互相挤压。
    fig.tight_layout()
    # 把已排版的图保存成图片文件；不需要显示器或手动关闭图形窗口。
    fig.savefig(path)
    # 关闭绘图对象并释放内存；保存图片后应清理，避免多次运行积累窗口。
    plt.close(fig)

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
