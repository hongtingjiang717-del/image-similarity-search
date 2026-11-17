# ------------ 数据路径与预处理配置 ------------
IMG_PATH = "../common/dataset/"  # 原始图像存储根目录（需确保存在子目录）
IMG_HEIGHT = 64
IMG_WIDTH = 64

# ------------ 随机性与数据划分配置 ------------
SEED = 42                         # 全局随机种子（确保实验可复现性）
TRAIN_RATIO = 0.70                # 训练集划分比例（75%训练，25%验证）
VAL_RATIO = 0.15       # 验证集比例（自动计算，无需修改）
TEST_RATIO = 0.15

# ------------ 训练超参数配置 ------------
LEARNING_RATE = 1e-3              # 初始学习率（AdamW优化器使用）
EPOCHS = 15                    # 总训练轮次（需平衡过拟合与欠拟合）
TRAIN_BATCH_SIZE = 32             # 训练批次大小（GPU显存不足时可调小）
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32              # 验证/测试批次大小（建议与训练批次一致）

CHROMA_INSERT_BATCH_SIZE = 5000  #每次输入chroma库的向量的数量



# ------------ 模型配置 ------------
PACKAGE_NAME = "image_similarity"
ENCODER_MODEL_NAME = "deep_encoder.pt"    # 编码器权重保存路径（需写权限）
DECODER_MODEL_NAME = "deep_decoder.pt"    # 解码器权重保存路径（需写权限）

CHROAM_BACKEND_PATH = "chroma_backend"  #后端存储的地址