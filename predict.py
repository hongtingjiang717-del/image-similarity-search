# 【初学者阅读说明】
# 这是单张图片的命令行入口；它调用 InferenceService，不训练模型。
# 示例：python predict.py classify common/dataset/0.jpg
# denoise 保存图片；classify 打印类别/分数；similar 打印相似文件名/距离。

"""单张图片 CPU 推理：python predict.py --help。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 json，把字典/列表与 JSON 文本相互转换
import json
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 从 PIL 导入 Image，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image
# 从 common.inference 导入 InferenceService；函数/类在对应文件实现，本文件直接复用。
from common.inference import InferenceService
# 从 common.paths 导入 ARTIFACT_DIR, DEFAULT_DEVICE；函数/类在对应文件实现，本文件直接复用。
from common.paths import ARTIFACT_DIR, DEFAULT_DEVICE

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记命令行参数 task；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 choices 限制允许输入的取值。
    parser.add_argument("task", choices=["denoise", "classify", "similar"])
    # 登记命令行参数 image；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。
    parser.add_argument("image", type=Path)
    # 登记命令行参数 --artifact-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    # 登记命令行参数 --device；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    # 登记命令行参数 --db-path；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。
    parser.add_argument("--db-path", type=Path)
    # 登记命令行参数 --output；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--output", type=Path, default=Path("denoised.png"))
    # 登记命令行参数 --add-noise；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 store_true 表示写出开关就为 True，不写为 False。
    parser.add_argument("--add-noise", action="store_true", help="额外加高斯噪声，演示去噪")
    # 登记命令行参数 --count；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--count", type=int, default=5)
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 创建推理服务，登记模型目录、设备和数据库路径；此时尚不加载所有权重。
    service = InferenceService(args.artifact_dir, args.device, args.db_path)
    # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
    with Image.open(args.image) as source:
        # 把打开的图片转换成 RGB 三通道，避免灰度图或带透明通道图片与模型输入不匹配。
        image = source.convert("RGB")
    # 条件分支：只有 args.task == 'denoise' 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if args.task == "denoise":
        # 去噪返回（模型输入图，恢复图）；_ 表示有意不用第一项，只保存恢复结果。
        _, output = service.denoise(image, args.add_noise)
        # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 把恢复后的 PIL 图片写到终端指定的目标路径，格式通常根据后缀决定。
        output.save(args.output)
        # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
        print(args.output.resolve())
    # else：前面的对应条件不成立（或 try 无异常）时，执行这个分支。
    else:
        # 按任务选择分类或检索；条件表达式只执行被选中的一侧，不会同时运行两个模型。
        result = service.classify(image) if args.task == "classify" else service.similar(image, args.count)
        # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
        print(json.dumps(result, ensure_ascii=False, indent=2))

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
