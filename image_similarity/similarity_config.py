# 【初学者阅读说明】
# 配置文件把常用参数集中保存；其他模块 import 后按“模块.变量名”读取。
# 大写变量名表示约定不随意改动的配置常量，Python 并不会强制禁止修改。
# 这里只做赋值，不会因为导入配置就开始训练或生成权重。

"""image_similarity 配置，默认 CPU、64×64 输入。"""
# 从 common.paths 导入 ARTIFACT_DIR, DATA_DIR, LABELS_PATH；函数/类在对应文件实现，本文件直接复用。
from common.paths import ARTIFACT_DIR, DATA_DIR, LABELS_PATH

# 模块包名字符串，用来说明配置属于哪个任务。
PACKAGE_NAME = "image_similarity"
# 图片数据目录；实际路径由 common.paths 统一提供。
IMG_PATH = DATA_DIR
# 输入图片高度和宽度（像素）；本项目模型按 64×64 工作。
IMG_HEIGHT = IMG_WIDTH = 64
# 随机种子：使实验中常用随机序列便于重复，不是类别标签。
SEED = 42
# 训练集比例，例如 0.75 表示约 75% 的图片用于训练。
TRAIN_RATIO = 0.75
# 验证集比例：用 1 减去训练比例得到。
VAL_RATIO = 1 - TRAIN_RATIO
# 学习率：控制优化器调整参数的步幅；1e-3 是 0.001。
LEARNING_RATE = 1e-3
# 训练总轮数：一轮指遍历一次训练集。
EPOCHS = 30
# 批次大小：一次交给模型计算多少张图；不是数据集总数。
TRAIN_BATCH_SIZE = TEST_BATCH_SIZE = VAL_BATCH_SIZE = FULL_BATCH_SIZE = 32
# 本模块训练结果的默认目录，Path 的 / 运算用于拼接目录。
OUTPUT_DIR = ARTIFACT_DIR / "similarity"
# 保存模型权重的文件名；这只是字符串，此处不会自动写文件。
ENCODER_MODEL_NAME = "deep_encoder.pt"
# 保存模型权重的文件名；这只是字符串，此处不会自动写文件。
DECODER_MODEL_NAME = "deep_decoder.pt"
# 把输出目录与文件名拼成完整路径；其他脚本按相同约定保存或加载。
ENCODER_MODEL_PATH = OUTPUT_DIR / ENCODER_MODEL_NAME
# 把输出目录与文件名拼成完整路径；其他脚本按相同约定保存或加载。
DECODER_MODEL_PATH = OUTPUT_DIR / DECODER_MODEL_NAME
# 把输出目录与文件名拼成完整路径；其他脚本按相同约定保存或加载。
EMBEDDING_NAME = OUTPUT_DIR / "embeddings.npy"
# 向量数据库在硬盘上的保存目录。
CHROMA_BACKEND_PATH = ARTIFACT_DIR / "chroma_backend"
# Chroma 集合名称，类似数据库里存放这一类记录的表名。
COLLECTION_NAME = "image_similarity"
# 编码和写库的期望批量；还要遵守客户端允许的最大批量。
CHROMA_INSERT_BATCH = 128  # 同时还会遵守当前 Chroma 实际允许的最大批量
