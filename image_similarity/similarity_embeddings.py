# 【初学者阅读说明】
# 本文件负责“把图片特征存起来”和“按特征寻找相似图片”。
# 建库：图片 → 同一编码器 → [B,512] → L2 归一化 → Chroma 持久化集合。
# 查询：新图片 → 同一套预处理/编码器/归一化 → 余弦距离最近的若干记录。
# id 直接用真实文件名，例如 15.jpg；metadata 另存 filename，前端不用猜图片路径。
# 模型指纹是根据权重计算的摘要，用来防止新模型查询旧模型生成的向量。
# 注意：--rebuild 会替换指定数据库中的本项目集合，默认不覆盖已有集合。

"""显式计算图片向量再写入 Chroma，不依赖自动文本嵌入或模型下载。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 hashlib，按模型参数计算 SHA-256 摘要，检查建库和查询是否使用相同模型
import hashlib
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 导入 numpy，处理数值数组、保存或读取 npy 向量文件；as np 给它起较短别名
import numpy as np
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 导入 torch.nn.functional，使用张量、神经网络计算及自动求导；as F 给它起较短别名
import torch.nn.functional as F
# 从 torch.utils.data 导入 DataLoader，用于使用张量、神经网络计算及自动求导。
from torch.utils.data import DataLoader
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.paths 导入 DEFAULT_DEVICE；函数/类在对应文件实现，本文件直接复用。
from common.paths import DEFAULT_DEVICE
# 从 common.utils 导入 get_device, load_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import get_device, load_weights
# 包内相对导入：前面的点表示当前包；从 similarity_data 取出 ImageDataset，避免误找同名顶层模块。
from .similarity_data import ImageDataset
# 包内相对导入：前面的点表示当前包；从 similarity_model 取出 ConvEncoder，避免误找同名顶层模块。
from .similarity_model import ConvEncoder
# 包内相对导入：前面的点表示当前包；从 本包 取出 similarity_config as config，避免误找同名顶层模块。
from . import similarity_config as config

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["create_embeddings", "get_embedding_collection", "search_similar_img_ids", "search_similar_images"]

# 依次读取各层参数的名字和数值，生成固定长度摘要；用于比较是否是同一套权重。
def model_fingerprint(encoder):
    # 创建 SHA-256 摘要计算器；它将许多参数压成一个便于比较的指纹字符串。
    digest = hashlib.sha256()
    # for 循环逐项遍历 sorted(encoder.state_dict().items())，每轮把当前元素赋给 (name, tensor)，再执行缩进内操作。
    for name, tensor in sorted(encoder.state_dict().items()):
        # 把参数名字编码成字节并加入摘要，防止只比较数值而忽略参数所属层。
        digest.update(name.encode())
        # 把参数移到 CPU、整理为连续数组、取原始字节，再加入指纹；不改变模型参数。
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    # 返回十六进制形式的摘要字符串，用于之后检查模型权重是否相同。
    return digest.hexdigest()

# 延迟导入 chromadb 并打开指定目录的持久化数据库；数据会保留在硬盘中。
def _client(db_path):
    # 导入 chromadb，操作保存在本地硬盘中的向量数据库
    import chromadb
    # 把 chromadb.PersistentClient(path=str(Path(db_path).resolve())) 交还给调用方，并结束本次函数调用；return 不会打印结果。
    return chromadb.PersistentClient(path=str(Path(db_path).resolve()))

# 验证集合是否建完，以及其中向量是否由当前编码器生成；不符合就抛异常。
def _check_collection(collection, encoder):
    # 若集合没有元数据就用空字典；or 会在左侧是假值时选择右侧。
    metadata = collection.metadata or {}
    # 读取集合状态或模型指纹并核对；不符合本次查询条件时拒绝继续。
    if metadata.get("status") != "ready":
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("图片索引未建完，请重新建库。")
    # 读取集合状态或模型指纹并核对；不符合本次查询条件时拒绝继续。
    if metadata.get("encoder_fingerprint") != model_fingerprint(encoder):
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("当前编码器与建库编码器不一致，请使用正确权重或 --rebuild 重建索引。")

# 分批读图、计算向量、写入数据库；成功后返回 collection 集合对象。
# encoder 是已加载权重的编码器；rebuild=False 默认保护已有集合。
def create_embeddings(encoder, data_dir=config.IMG_PATH, db_path=config.CHROMA_BACKEND_PATH,
                      batch_size=config.CHROMA_INSERT_BATCH, device=None, rebuild=False):
    """流式编码并入库；已有集合需显式 --rebuild，避免残留旧图片和旧模型向量。"""
    # 检查数值参数的合法范围，不允许负噪声强度或非正的批次/结果数量。
    if batch_size < 1:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("batch_size 必须为正整数")
    # 创建统一图像预处理：缩放尺寸、转成 [C,H,W] 浮点张量。
    dataset = ImageDataset(data_dir, build_transform(config.IMG_HEIGHT, config.IMG_WIDTH))
    # 文件名（包括后缀）直接作为 id，兼容缺号、非数字文件名和多种图片格式。
    # 打开或创建指定位置的持久化数据库客户端。
    client = _client(db_path)
    # try 包住可能失败的操作；只有发生匹配异常时才进入后面的 except，正常时继续后续流程。
    try:
        # 尝试读取同名集合，判断索引是否已经存在；查不到会进入指定异常分支。
        client.get_collection(config.COLLECTION_NAME, embedding_function=None)
    # 捕获指定异常；as exc 把异常对象保存为 exc；不匹配的异常不会由这个分支处理。
    except Exception as exc:
        # 从 chromadb.errors 导入 NotFoundError，用于操作保存在本地硬盘中的向量数据库。
        from chromadb.errors import NotFoundError
        # 条件分支：只有 not isinstance(exc, NotFoundError) 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
        if not isinstance(exc, NotFoundError):
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise
    # else：前面的对应条件不成立（或 try 无异常）时，执行这个分支。
    else:
        # 条件分支：只有 not rebuild 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
        if not rebuild:
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise FileExistsError("索引集合已存在；确认需要替换时使用 --rebuild。")
        # 已确认 rebuild=True，删除该项目原集合；接下来重建，不会把新旧权重向量混在一起。
        client.delete_collection(config.COLLECTION_NAME)
    # 记录这次用来编码图片的完整权重指纹；以后查询要与它一致。
    fingerprint = model_fingerprint(encoder)
    # 登记模型指纹、building 状态、输入尺寸、来源目录和图片总数；先标记尚未建完。
    metadata = dict(encoder_fingerprint=fingerprint, status="building", image_size="64x64",
                    source_dir=str(Path(data_dir).resolve()), count=len(dataset))
    # 创建图片向量集合，明确使用 cosine 余弦距离；embedding_function=None 表示由本项目自己提供向量。
    collection = client.create_collection(
        config.COLLECTION_NAME, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}}, metadata=metadata)
    # 取配置批量与 Chroma 当前允许最大批量的较小值，避免超出写入限制。
    actual_batch = min(batch_size, client.get_max_batch_size())
    # 创建按批取样器；batch_size 控制每批张数，shuffle 决定是否打乱，num_workers 决定读图子进程数。
    loader = DataLoader(dataset, batch_size=actual_batch, shuffle=False, num_workers=0)
    # 把设备字符串转成 PyTorch 设备对象；CPU 与 GPU 上的模型和张量要匹配。
    device = torch.device(device) if device is not None else next(encoder.parameters()).device
    # 将编码器移动到设备并切为评估模式；建库不是继续训练。
    encoder.to(device).eval()
    # 用偏移量记录已处理图片数，让每批向量与相应文件名严格对齐。
    offset = 0
    # with torch.no_grad() 临时关闭梯度记录，退出块后恢复原状态；它不等同于 model.eval()。
    with torch.no_grad():
        # 每次从加载器取一批（输入，目标）并解包；一轮循环会处理完所有批次。
        for images, _ in loader:
            # 编码为 [B,512]，从第 1 维展平，按每张图做 L2 归一化，再转 CPU NumPy 数组供 Chroma 使用。
            vectors = F.normalize(encoder(images.to(device)).flatten(1), dim=1).cpu().numpy()
            # 检查数值是否正常且向量非零，避免余弦距离计算遇到无效特征。
            if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < 1e-12):
                # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                raise ValueError("编码器产生无效或全零向量，不能构建余弦索引。")
            # 取出恰好与当前批次对应的文件路径切片，尾批按实际张数处理。
            paths = dataset.paths[offset:offset + len(images)]
            # 批量写入显式向量、真实文件名 id 和元数据；同 id 存在时更新，不用 Chroma 自动生成文本嵌入。
            collection.upsert(ids=[p.name for p in paths], embeddings=vectors,
                              metadatas=[{"filename": p.name} for p in paths])
            # 累计已处理的图片数，下一批从新的偏移量继续取文件名。
            offset += len(images)
            # 打印当前进度、形状或结果，便于观察程序；print 本身不会训练或保存模型。
            print(f"索引写入：{offset}/{len(dataset)}", flush=True)
    # 所有批次都成功写完后，才把状态从 building 改为 ready。
    metadata["status"] = "ready"
    # 把完成状态持久化到集合；中途中断时就不会误把半成品当完整索引。
    collection.modify(metadata=metadata)
    # 返回可用的 Chroma 集合对象，调用者可据此计数或查询。
    return collection

# 只打开已有的合格图片集合；没有数据库或集合时提示先建库。
def get_embedding_collection(encoder, db_path=config.CHROMA_BACKEND_PATH):
    # 查询只获取已有集合，不静默创建空库。
    # 先检查必要的目录/文件是否存在；不存在时给出可理解的提示。
    if not Path(db_path).is_dir():
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise FileNotFoundError(f"向量数据库不存在：{db_path}，请先运行建库命令。")
    # 打开或创建指定位置的持久化数据库客户端。
    client = _client(db_path)
    # try 包住可能失败的操作；只有发生匹配异常时才进入后面的 except，正常时继续后续流程。
    try:
        # 获取已有图片集合；不给自动嵌入函数，查询时将显式传入计算好的向量。
        collection = client.get_collection(config.COLLECTION_NAME, embedding_function=None)
    # 捕获指定异常；as exc 把异常对象保存为 exc；不匹配的异常不会由这个分支处理。
    except Exception as exc:
        # 从 chromadb.errors 导入 NotFoundError，用于操作保存在本地硬盘中的向量数据库。
        from chromadb.errors import NotFoundError
        # 条件分支：只有 isinstance(exc, NotFoundError) 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
        if isinstance(exc, NotFoundError):
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise FileNotFoundError("数据库中没有图片集合，请先运行建库命令。") from exc
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise
    # 查询前检查集合完成状态与模型指纹，避免模型和索引不匹配。
    _check_collection(collection, encoder)
    # 返回可用的 Chroma 集合对象，调用者可据此计数或查询。
    return collection

# 输入集合、单张图片张量、编码器、返回数量；输出按距离排序的结果字典列表。
# 余弦距离越小表示这里定义的特征越相似，它不是商品类别概率。
def search_similar_images(collection, image_tensor, encoder, img_cnt=5):
    # 检查数值参数的合法范围，不允许负噪声强度或非正的批次/结果数量。
    if img_cnt < 1:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("img_cnt 必须为正整数")
    # 查询前检查集合完成状态与模型指纹，避免模型和索引不匹配。
    _check_collection(collection, encoder)
    # 读取集合实际有多少条图片向量，用于限制返回数量。
    count = collection.count()
    # 没有可检索记录时返回空列表，避免发出无意义查询。
    if count == 0:
        # 用列表推导式整理输出，保持数据库返回的顺序；每一项是一个结果记录或文件名。
        return []
    # ndim 是维度数；三维 [C,H,W] 表示还没有批次轴。
    if image_tensor.ndim == 3:
        # 单图只有 [3,H,W] 时补上批次维，变成 [1,3,H,W]。
        image_tensor = image_tensor.unsqueeze(0)
    # 检查输入/输出的维度与大小是否符合当前网络约定；不匹配时提前提示，而不是让矩阵运算难懂地报错。
    if image_tensor.shape[0] != 1:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("一次只查询一张图片")
    # 切换评估模式；对 Dropout/BatchNorm 等层有效，但不会自动关闭梯度计算。
    encoder.eval()
    # with torch.no_grad() 临时关闭梯度记录，退出块后恢复原状态；它不等同于 model.eval()。
    with torch.no_grad():
        # 用同一编码器得到查询图的 [1,512] 特征，再按每行做 L2 归一化，与建库方式保持一致。
        vector = F.normalize(encoder(image_tensor.to(next(encoder.parameters()).device)).flatten(1), dim=1)
    # 把查询特征移到 CPU 并转换为 NumPy；数据库接口不直接接收 GPU 张量。
    vector = vector.cpu().numpy()
    # 检查数值是否正常且向量非零，避免余弦距离计算遇到无效特征。
    if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-12:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("查询图片产生无效或全零向量")
    # 查询余弦距离最近的记录，只返回文件元数据和距离；数量不超过集合实际条数。
    result = collection.query(query_embeddings=vector, n_results=min(img_cnt, count),
                              include=["metadatas", "distances"])
    # 用列表推导式整理输出，保持数据库返回的顺序；每一项是一个结果记录或文件名。
    return [{"id": image_id, "filename": meta["filename"], "distance": float(distance)}
            for image_id, meta, distance in zip(result["ids"][0], result["metadatas"][0], result["distances"][0])]

# 兼容需要 id 列表的调用方；只取检索结果中的真实文件名 id。
def search_similar_img_ids(collection, image_tensor, img_cnt=5, encoder=None):
    """返回真实文件名 id；显式传入编码器，禁止把图片直接当向量查询。"""
    # 条件分支：只有 encoder is None 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if encoder is None:
        # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
        raise ValueError("请传入与建库一致的 encoder")
    # 用列表推导式整理输出，保持数据库返回的顺序；每一项是一个结果记录或文件名。
    return [item["id"] for item in search_similar_images(collection, image_tensor, encoder, img_cnt)]

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description="使用已训练编码器构建 Chroma 图片索引")
    # 登记命令行参数 --encoder；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--encoder", type=Path, default=config.ENCODER_MODEL_PATH)
    # 登记命令行参数 --data-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--data-dir", type=Path, default=config.IMG_PATH)
    # 登记命令行参数 --db-path；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--db-path", type=Path, default=config.CHROMA_BACKEND_PATH)
    # 登记命令行参数 --device；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    # 登记命令行参数 --batch-size；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--batch-size", type=int, default=config.CHROMA_INSERT_BATCH)
    # 登记命令行参数 --rebuild；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 store_true 表示写出开关就为 True，不写为 False。
    parser.add_argument("--rebuild", action="store_true", help="替换指定数据库中的本项目图片集合")
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 读取或选择当前计算设备，之后网络与输入都放到此处进行运算。
    device = get_device(args.device)
    # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
    encoder = load_weights(ConvEncoder(), args.encoder, device)
    # 调用完整建库流程，使用所选编码器、目录、批次、设备与覆盖开关。
    create_embeddings(encoder, args.data_dir, args.db_path, args.batch_size, device, args.rebuild)

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
