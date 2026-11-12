# 相似图片模型的训练入口，联合训练编码器和解码器。
# 训练结束后重新加载验证损失最低的编码器，导出向量及每行对应的真实文件名。
# npy 导出与 Chroma 建库是两个步骤；本文件不会自动构建网页查询所用数据库。

"""运行：python -m image_similarity.similarity_train --device cpu。"""
# 导入 json，把字典/列表与 JSON 文本相互转换
import json
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.training 导入 training_parser, make_loaders, fit；函数/类在对应文件实现，本文件直接复用。
from common.training import training_parser, make_loaders, fit
# 从 common.utils 导入 seed_everything, get_device, load_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import seed_everything, get_device, load_weights
# 包内相对导入：前面的点表示当前包；从 本包 取出 similarity_config as config，避免误找同名顶层模块。
from . import similarity_config as config
# 包内相对导入：前面的点表示当前包；从 similarity_data 取出 ImageDataset，避免误找同名顶层模块。
from .similarity_data import ImageDataset
# 包内相对导入：前面的点表示当前包；从 similarity_model 取出 ConvEncoder, ConvDecoder，避免误找同名顶层模块。
from .similarity_model import ConvEncoder, ConvDecoder
# 包内相对导入：前面的点表示当前包；从 similarity_engine 取出 train_step, val_step, create_embedding，避免误找同名顶层模块。
from .similarity_engine import train_step, val_step, create_embedding

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 建立本任务的命令行解析器，并注册数据目录、轮数、批次大小等通用选项。
    parser = training_parser(__doc__, config)
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 在随机初始化和划分数据之前固定随机源，提高同一环境中实验的可重复性。
    seed_everything(args.seed)
    # 读取或选择当前计算设备，之后网络与输入都放到此处进行运算。
    device = get_device(args.device)
    # 创建统一图像预处理：缩放尺寸、转成 [C,H,W] 浮点张量。
    dataset = ImageDataset(args.data_dir, build_transform(config.IMG_HEIGHT, config.IMG_WIDTH))
    # 按固定随机种子划分数据，并返回训练、验证和全量三个批次加载器。
    train, val, full = make_loaders(dataset, args, config.TRAIN_RATIO)
    # 建立这一组初始数据；列表用 []、元组常用 ()、集合用 {}，逗号分隔各项。
    encoder, decoder = ConvEncoder().to(device), ConvDecoder().to(device)
    # 创建 AdamW 优化器，登记要学习的参数与学习率；只有调用 step 时才更新参数。
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    # 开始完整训练：把模型、损失、优化器、训练/验证数据传给公共 fit，由它按轮调用引擎并保存最优参数。
    fit([encoder, decoder], [config.ENCODER_MODEL_NAME, config.DECODER_MODEL_NAME],
        train_step, val_step, train, val, torch.nn.MSELoss(), optimizer, args, device)
    # 导出必须使用验证最优权重，与之后的建库和查询保持一致。
    # 加载验证最优的权重，再做预览或导出；避免误用最后一轮可能较差的模型。
    load_weights(encoder, args.output_dir / config.ENCODER_MODEL_NAME, device)
    # 用最佳编码器导出全量加载器中每张图的特征，按原顺序保存。
    create_embedding(encoder, full, args.output_dir / "embeddings.npy", device)
    # 按导出向量的真实行顺序取图片文件名，避免向量和文件名错位。
    filenames = [dataset.paths[index].name for index in full.dataset.indices]
    # 将结果文本写入指定文件，encoding="utf-8" 保证中文正确；JSON 序列化让其他程序也能读取。
    (args.output_dir / "embedding_filenames.json").write_text(
        json.dumps(filenames, ensure_ascii=False, indent=2), encoding="utf-8")

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
