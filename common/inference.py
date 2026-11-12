# 【初学者阅读说明】
# 训练是调整参数；推理是拿已经调好的模型处理新图片。本文件只负责推理。
# 命令行 predict.py 与 Flask 后端都复用 InferenceService，避免两套处理方式不一致。
# 第一次请求某个任务才加载其权重，之后保存在 self.models 中重复使用。
# 输入：PIL 图片对象；输出：去噪图片、分类字典或相似商品列表。

"""命令行与 Web 共用推理服务；没有权重时明确报错，不用随机模型冒充结果。"""
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 从 PIL 导入 Image，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.paths 导入 ARTIFACT_DIR；函数/类在对应文件实现，本文件直接复用。
from common.paths import ARTIFACT_DIR
# 从 common.utils 导入 get_device, load_weights；函数/类在对应文件实现，本文件直接复用。
from common.utils import get_device, load_weights
# 从 image_denoising.denoising_model 导入 ConvDenoiser；函数/类在对应文件实现，本文件直接复用。
from image_denoising.denoising_model import ConvDenoiser
# 从 image_classification.classification_model 导入 Classifier；函数/类在对应文件实现，本文件直接复用。
from image_classification.classification_model import Classifier
# 从 image_classification.classification_config 导入 classification_names；函数/类在对应文件实现，本文件直接复用。
from image_classification.classification_config import classification_names
# 从 image_similarity.similarity_model 导入 ConvEncoder；函数/类在对应文件实现，本文件直接复用。
from image_similarity.similarity_model import ConvEncoder

# 定义推理服务类：类像功能模板，对象保存自己的模型缓存、路径和设备。
class InferenceService:
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self, artifact_dir=ARTIFACT_DIR, device="cpu", db_path=None):
        # 将这项初始化结果保存在 self.artifact_dir 中，供同一对象或测试类的后续方法使用。
        self.artifact_dir = Path(artifact_dir)
        # 将这项初始化结果保存在 self.device 中，供同一对象或测试类的后续方法使用。
        self.device = get_device(device)
        # 将这项初始化结果保存在 self.db_path 中，供同一对象或测试类的后续方法使用。
        self.db_path = Path(db_path) if db_path is not None else self.artifact_dir / "chroma_backend"
        # 准备模型缓存字典：任务名→已加载模型；第一次加载后重复使用。
        self.models = {}
        # 集合尚未打开；首次相似检索时才读取数据库并校验。
        self.collection = None
        # 训练与推理共用默认 64×64 预处理，避免模型看到不同输入格式。
        self.transform = build_transform()

    # 按任务名选择网络和权重路径；命中 self.models 缓存就不再重复读取硬盘。
    def _model(self, task):
        # 字典把每个任务映射到（模型类，权重相对路径）的二元组；只登记，不立刻实例化。
        definitions = {
            "denoising": (ConvDenoiser, "denoising/denoiser.pt"),
            "classification": (Classifier, "classification/classifier.pt"),
            "similarity": (ConvEncoder, "similarity/deep_encoder.pt"),
        }
        # not in 检查缓存里是否还没有这个任务；只在第一次使用时加载。
        if task not in self.models:
            # 按任务名从字典拿到（模型类，权重相对路径），并解包；cls() 会创建那个模型。
            cls, relative_path = definitions[task]
            # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
            self.models[task] = load_weights(cls(), self.artifact_dir / relative_path, self.device)
        # 返回缓存中指定任务的模型对象，不会每次都重新随机初始化。
        return self.models[task]

    # 输入一张 PIL 图；返回真正输入模型的图片与去噪输出，均为 PIL 图对象。
    # @torch.inference_mode() 是装饰器：整次调用关闭反向求导相关记录，用于推理而非训练。
    @torch.inference_mode()
    def denoise(self, image, add_noise=False, noise_factor=0.5):
        # 图片先转 [3,64,64]，unsqueeze(0) 加批次维成 [1,3,64,64]，再移动设备。
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # add_noise=True 才人为加高斯噪声；否则直接处理用户图像。randn_like 自动匹配形状和设备。
        noisy = (tensor + noise_factor * torch.randn_like(tensor)).clamp(0, 1) if add_noise else tensor
        # 先取去噪模型，再把带噪输入传给它；两个括号分别是“取模型”和“调用模型”。
        output = self._model("denoising")(noisy)
        # [0] 取批次中的唯一图片，去掉批次维；返回输入图和恢复图两个 PIL 对象。
        return self.to_image(noisy[0]), self.to_image(output[0])

    # 输入一张 PIL 图；返回 label、中文 name、confidence 三个字段。
    # @torch.inference_mode() 是装饰器：整次调用关闭反向求导相关记录，用于推理而非训练。
    @torch.inference_mode()
    def classify(self, image):
        # 图片先转 [3,64,64]，unsqueeze(0) 加批次维成 [1,3,64,64]，再移动设备。
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        # 得到 [1,5] 类别分数，沿类别维 softmax，取第一个样本，得到五个和为 1 的数。
        probabilities = self._model("classification")(tensor).softmax(dim=1)[0]
        # argmax 找最大分数所在下标；item 取 Python 数值，int 转类别整数。
        label = int(probabilities.argmax().item())
        # 返回字典，调用方可按键取得结果；花括号里的键名就是对外的数据字段。
        return {"label": label, "name": classification_names[label],
                "confidence": float(probabilities[label].item())}

    # 输入图片与希望返回的数量；返回真实文件名和向量距离组成的列表。
    # @torch.inference_mode() 是装饰器：整次调用关闭反向求导相关记录，用于推理而非训练。
    @torch.inference_mode()
    def similar(self, image, count=5):
        # 从 image_similarity.similarity_embeddings 导入 get_embedding_collection, search_similar_images；函数/类在对应文件实现，本文件直接复用。
        from image_similarity.similarity_embeddings import get_embedding_collection, search_similar_images
        # 取出已训练编码器；第一次需要从权重文件加载，以后直接取缓存模型。
        encoder = self._model("similarity")
        # 数据库集合还没打开时才读取；后续请求复用同一个集合对象。
        if self.collection is None:
            # 将这项初始化结果保存在 self.collection 中，供同一对象或测试类的后续方法使用。
            self.collection = get_embedding_collection(encoder, self.db_path)
        # 把 search_similar_images(self.collection, self.transform(image.convert('RGB')), encoder, count) 交还给调用方，并结束本次函数调用；return 不会打印结果。
        return search_similar_images(self.collection, self.transform(image.convert("RGB")), encoder, count)

    # 把 [3,H,W] 的浮点张量转换为 [H,W,3] 的 0～255 无符号整数图片。
    # 这是 staticmethod：不使用 self，不依赖某一个服务对象的状态。
    # @staticmethod 表示静态方法：调用时不会自动传入 self。
    @staticmethod
    def to_image(tensor):
        # 从求导图分离、移到 CPU、限制像素范围，把 [C,H,W] 改为 [H,W,C] 后转 NumPy。
        array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        # 像素乘 255、四舍五入、转 0～255 无符号字节，再由数组生成彩色图片。
        return Image.fromarray((array * 255).round().astype("uint8"))
