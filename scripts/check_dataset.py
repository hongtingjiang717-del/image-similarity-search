# 【初学者阅读说明】
# 本脚本检查现有图片与标签能否匹配，不训练模型、不修改数据。
# 普通运行检查目录与标签；加 --verify-images 才逐张解码，检查图片是否损坏。

"""检查图片与 CSV 的对应关系，可选择完整解码以检查损坏文件。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 json，把字典/列表与 JSON 文本相互转换
import json
# 从 collections 导入 Counter，用于使用 Counter 统计各类别出现次数。
from collections import Counter
# 从 PIL 导入 Image，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image
# 从 common.paths 导入 DATA_DIR, LABELS_PATH；函数/类在对应文件实现，本文件直接复用。
from common.paths import DATA_DIR, LABELS_PATH
# 从 image_classification.classification_data 导入 ImageDataset；函数/类在对应文件实现，本文件直接复用。
from image_classification.classification_data import ImageDataset

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记命令行参数 --data-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--data-dir", default=DATA_DIR)
    # 登记命令行参数 --labels；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--labels", default=LABELS_PATH)
    # 登记命令行参数 --verify-images；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 store_true 表示写出开关就为 True，不写为 False。
    parser.add_argument("--verify-images", action="store_true")
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
    dataset = ImageDataset(args.data_dir, args.labels)
    # 收集打不开或不能完整解码的图片；只有选择完整验证时才逐图检查。
    failures = []
    # 只有要求完整检查时才逐张解码，普通标签检查不做这一步。
    if args.verify_images:
        # for 循环逐项遍历 dataset.paths，每轮把当前元素赋给 path，再执行缩进内操作。
        for path in dataset.paths:
            # try 包住可能失败的操作；只有发生匹配异常时才进入后面的 except，正常时继续后续流程。
            try:
                # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
                with Image.open(path) as image:
                    # 强制把整张图的像素读出来；仅 Image.open 可能只读取文件头，不能充分发现损坏。
                    image.load()
            # 捕获指定异常；as exc 把异常对象保存为 exc；不匹配的异常不会由这个分支处理。
            except Exception as exc:
                # 将本次处理的结果追加到列表末尾，保留先前已经收集的数据。
                failures.append({"file": path.name, "error": str(exc)})
    # 把样本数、损失或检查统计值组织为结果字典，随后转成 JSON 方便阅读。
    result = dict(images=len(dataset), labels=len(dataset.label_dict),
                  targets=dict(Counter(dataset.targets)), invalid_images=failures,
                  image_decode_checked=args.verify_images)
    # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 条件分支：只有 failures 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if failures:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise SystemExit(1)

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
