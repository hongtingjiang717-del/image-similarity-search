# 【初学者阅读说明】
# 本文件只决定“文件放在哪里”和“默认用什么设备”，不读取图片、不训练模型。
# 数据流：环境变量或默认值 → Path 路径对象 → 各模块的 config 文件。
# Path 是路径工具；/ 在 Path 对象之间表示拼接路径，不是数学除法。
# 环境变量是启动程序前设置的外部配置；本项目未设置时默认使用项目内的目录和 CPU。

"""所有路径相对项目位置解析；环境变量可用于迁移数据和训练产物。"""
# 导入 os，读取环境变量或进行旧实验中的目录操作
import os
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path

# __file__ 是当前源码文件路径；resolve 得绝对路径，两次 parent 回到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 先用 os.getenv 读取外部指定的路径，未设置就采用项目内默认路径；expanduser 展开用户目录，resolve 转绝对路径。
DATA_DIR = Path(os.getenv("IMAGE_SIM_DATA_DIR", str(PROJECT_ROOT / "common/dataset"))).expanduser().resolve()
# 先用 os.getenv 读取外部指定的路径，未设置就采用项目内默认路径；expanduser 展开用户目录，resolve 转绝对路径。
LABELS_PATH = Path(os.getenv("IMAGE_SIM_LABELS_PATH", str(PROJECT_ROOT / "common/fashion-labels.csv"))).expanduser().resolve()
# 先用 os.getenv 读取外部指定的路径，未设置就采用项目内默认路径；expanduser 展开用户目录，resolve 转绝对路径。
ARTIFACT_DIR = Path(os.getenv("IMAGE_SIM_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts"))).expanduser().resolve()
# 当前默认 CPU；以后可在命令行传 --device cuda 或设置此变量。
# 读取默认设备环境变量，没有设置时用字符串 cpu；不自动假定本机有显卡。
DEFAULT_DEVICE = os.getenv("IMAGE_SIM_DEVICE", "cpu")
