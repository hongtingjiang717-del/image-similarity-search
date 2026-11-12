# 【初学者阅读说明】
# 测试不是业务功能，它用小型可控输入检查项目有没有被改坏。
# unittest 自动运行 test_ 开头的方法；assert... 表示实际结果必须满足某个条件。
# 本套集成测试按 test_01→test_02→test_03 运行：检查结构→训练临时模型→建库/请求接口。
# 样本由随机数组生成，不是真实商品测试集；所有权重和数据库写到临时目录。
# setUpClass 在测试组开始前准备共享数据，tearDownClass 在全部测试后清理。

"""使用临时小数据集覆盖训练、存取、Chroma 和 HTTP，不依赖真实训练权重。"""
# 导入 csv，按 CSV 的行与列读取/写入分类标签
import csv
# 导入 io，在内存中读写字节或文本，不必先创建硬盘文件
import io
# 导入 tempfile，创建测试专用临时目录
import tempfile
# 导入 unittest，运行自动化测试，并用断言检查结果
import unittest
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 导入 numpy，处理数值数组、保存或读取 npy 向量文件；as np 给它起较短别名
import numpy as np
# 导入 torch，使用张量、神经网络计算及自动求导
import torch
# 从 PIL 导入 Image，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image
# 从 common.data 导入 build_transform；函数/类在对应文件实现，本文件直接复用。
from common.data import build_transform
# 从 common.utils 导入 load_weights, seed_everything；函数/类在对应文件实现，本文件直接复用。
from common.utils import load_weights, seed_everything
# 从 image_denoising.denoising_data 导入 ImageDataset as DenoisingDataset；函数/类在对应文件实现，本文件直接复用。
from image_denoising.denoising_data import ImageDataset as DenoisingDataset
# 从 image_denoising.denoising_model 导入 ConvDenoiser；函数/类在对应文件实现，本文件直接复用。
from image_denoising.denoising_model import ConvDenoiser
# 从 image_classification.classification_data 导入 ImageDataset as ClassificationDataset；函数/类在对应文件实现，本文件直接复用。
from image_classification.classification_data import ImageDataset as ClassificationDataset
# 从 image_classification.classification_model 导入 Classifier；函数/类在对应文件实现，本文件直接复用。
from image_classification.classification_model import Classifier
# 从 image_similarity.similarity_model 导入 ConvEncoder, ConvDecoder；函数/类在对应文件实现，本文件直接复用。
from image_similarity.similarity_model import ConvEncoder, ConvDecoder

# 继承 unittest.TestCase，得到 assertEqual、assertRaises 等测试工具。
class ProjectIntegrationTest(unittest.TestCase):
    # 类方法中的 cls 表示测试类；创建整组测试共享的临时目录、图片和标签。
    # @classmethod 表示类方法：第一个参数 cls 接收类本身。
    @classmethod
    def setUpClass(cls):
        # 限制测试使用的 CPU 计算线程数量，让小张量测试更稳定且避免过度并行开销。
        torch.set_num_threads(2)
        # 在随机初始化和划分数据之前固定随机源，提高同一环境中实验的可重复性。
        seed_everything(42)
        # 将这项初始化结果保存在 cls.temporary 中，供同一对象或测试类的后续方法使用。
        cls.temporary = tempfile.TemporaryDirectory()
        # 将这项初始化结果保存在 cls.root 中，供同一对象或测试类的后续方法使用。
        cls.root = Path(cls.temporary.name)
        # 将这项初始化结果保存在 cls.data 中，供同一对象或测试类的后续方法使用。
        cls.data = cls.root / "images"
        # 创建目录；parents=True 会连同缺少的父目录创建，exist_ok=True 允许目录已经存在。
        cls.data.mkdir()
        # 刻意使用不连续文件名，验证不能用样本索引充当 CSV id。
        # 将这项初始化结果保存在 cls.ids 中，供同一对象或测试类的后续方法使用。
        cls.ids = [2, 8, 15, 21, 30, 44, 51, 79]
        # 创建独立 NumPy 随机数发生器；固定种子便于生成可重复的测试图片。
        rng = np.random.default_rng(42)
        # for 循环逐项遍历 cls.ids，每轮把当前元素赋给 image_id，再执行缩进内操作。
        for image_id in cls.ids:
            # 生成 72×80 RGB 随机像素图并保存；integers 上限 256 不包含 256，因此像素为 0～255。
            Image.fromarray(rng.integers(0, 256, (72, 80, 3), dtype=np.uint8)).save(cls.data / f"{image_id}.jpg")
        # 将结果文本写入指定文件，encoding="utf-8" 保证中文正确；JSON 序列化让其他程序也能读取。
        (cls.data / "readme.txt").write_text("忽略非图片文件", encoding="utf-8")
        # 将这项初始化结果保存在 cls.labels 中，供同一对象或测试类的后续方法使用。
        cls.labels = cls.root / "labels.csv"
        # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
        with cls.labels.open("w", newline="", encoding="utf-8") as stream:
            # 创建 CSV 写入器，负责正确处理分隔符和行格式。
            writer = csv.writer(stream)
            # 按 CSV 格式写入表头或数据行；逗号和换行由 csv 模块处理。
            writer.writerow(["id", "target"])
            # 按 CSV 格式写入表头或数据行；逗号和换行由 csv 模块处理。
            writer.writerows((image_id, i % 5) for i, image_id in enumerate(cls.ids))
        # 将这项初始化结果保存在 cls.artifacts 中，供同一对象或测试类的后续方法使用。
        cls.artifacts = cls.root / "artifacts"

    # 整组测试完成后清理临时目录；Windows 文件被占用时保留临时文件等待进程退出。
    # @classmethod 表示类方法：第一个参数 cls 接收类本身。
    @classmethod
    def tearDownClass(cls):
        # Windows 上 Chroma 的文件句柄可能持续到进程结束；只在临时目录生成测试产物。
        # try 包住可能失败的操作；只有发生匹配异常时才进入后面的 except，正常时继续后续流程。
        try:
            # 删除临时目录及其测试产物；这里的路径由 tempfile 创建，不是正式数据目录。
            cls.temporary.cleanup()
        # 只处理这里指定的异常类型；其他错误仍向上抛出，不一律忽略。
        except PermissionError:
            # pass 是合法的空语句；这里表示继承父类实现或按已有约定不再执行额外动作，不是网络计算。
            pass

    # 用内存字节流构造一次上传请求的数据，避免测试依赖用户手动选图。
    def upload(self):
        # 返回字典，调用方可按键取得结果；花括号里的键名就是对外的数据字段。
        return {"image": (io.BytesIO((self.data / "2.jpg").read_bytes()), "2.jpg")}

    # 检查图片标签映射、噪声可重复性，以及三个网络的输入输出形状。
    def test_01_dataset_and_shapes(self):
        # 创建数据集对象并准备图片名册；具体像素通常到 dataset[idx] 被调用时才读取。
        dataset = ClassificationDataset(self.data, self.labels)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(len(dataset), 8)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(dataset[1][1], 1)
        # 建立固定噪声的临时去噪数据集，接下来检查同一图片重复读取是否得到相同噪声。
        denoise = DenoisingDataset(self.data, noise_seed=42)
        # 断言这个条件必须为真，例如文件应已生成、固定种子的噪声应一致。
        self.assertTrue(torch.equal(denoise[0][0], denoise[0][0]))
        # 生成 [0,1) 的随机浮点张量，作为测试输入；它不是来自真实图片的像素。
        inputs = torch.rand(1, 3, 64, 64)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(tuple(ConvDenoiser()(inputs).shape), (1, 3, 64, 64))
        # 测试要求下方代码必须抛出指定异常；若没有抛出，说明保护逻辑失效，测试会失败。
        with self.assertRaises(ValueError):
            # 在 assertRaises 块里故意传入不支持的 68×68，检查模型是否提前给出 ValueError。
            ConvDenoiser()(torch.rand(1, 3, 68, 68))
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(tuple(Classifier()(inputs).shape), (1, 5))
        # 执行当前网络或当前层的前向运算，把结果保存为下一步要使用的张量。
        encoded = ConvEncoder()(inputs)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(tuple(encoded.shape), (1, 512))
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(tuple(ConvDecoder()(encoded).shape), (1, 3, 64, 64))

    # 分别运行三类训练入口，检查保存权重、对比图和向量文件。
    def test_02_train_all_modules(self):
        # 从 image_denoising.denoising_train 导入 main as denoise_main；函数/类在对应文件实现，本文件直接复用。
        from image_denoising.denoising_train import main as denoise_main
        # 从 image_classification.classification_train 导入 main as classify_main；函数/类在对应文件实现，本文件直接复用。
        from image_classification.classification_train import main as classify_main
        # 从 image_similarity.similarity_train 导入 main as similarity_main；函数/类在对应文件实现，本文件直接复用。
        from image_similarity.similarity_train import main as similarity_main
        # 建立这一组初始数据；列表用 []、元组常用 ()、集合用 {}，逗号分隔各项。
        common = ["--data-dir", str(self.data), "--device", "cpu", "--epochs", "1", "--batch-size", "3"]
        # 列表 + 用于拼接参数；实际调用去噪入口训练一轮，把模型和预览写到临时目录。
        denoise_main(common + ["--output-dir", str(self.artifacts / "denoising"), "--preview"])
        # 真实调用分类训练入口，明确使用临时标签和输出目录，避免碰到正式权重。
        classify_main(common + ["--labels", str(self.labels), "--output-dir", str(self.artifacts / "classification")])
        # 真实调用相似自编码器训练入口，训练和导出结果都放在临时目录。
        similarity_main(common + ["--output-dir", str(self.artifacts / "similarity")])
        # 断言这个条件必须为真，例如文件应已生成、固定种子的噪声应一致。
        self.assertTrue((self.artifacts / "denoising/preview.png").is_file())
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(np.load(self.artifacts / "similarity/embeddings.npy").shape, (8, 512))
        # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
        model = load_weights(Classifier(), self.artifacts / "classification/classifier.pt")
        # 断言这个条件必须为假；这里检查加载的模型没有留在训练模式。
        self.assertFalse(model.training)

    # 检查 Chroma 检索、错误分支、网页接口、命令行推理和验证集评估。
    def test_03_index_and_http(self):
        # 从 image_similarity.similarity_embeddings 导入 create_embeddings, get_embedding_collection, search_similar_images；函数/类在对应文件实现，本文件直接复用。
        from image_similarity.similarity_embeddings import create_embeddings, get_embedding_collection, search_similar_images
        # 从 web.web_app 导入 create_app；函数/类在对应文件实现，本文件直接复用。
        from web.web_app import create_app
        # 创建或取用相同结构的模型，读入已训练参数，并切换到评估模式。
        encoder = load_weights(ConvEncoder(), self.artifacts / "similarity/deep_encoder.pt")
        # 用 Path 的 / 运算拼接出本次测试数据库目录，与正式 artifacts 隔离。
        db = self.artifacts / "chroma_backend"
        # 每批 3 张对临时图片建库；8 张图会形成 3、3、2 三批，覆盖尾批处理。
        collection = create_embeddings(encoder, self.data, db, batch_size=3)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(collection.count(), 8)
        # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
        with Image.open(self.data / "2.jpg") as image:
            # 创建统一图像预处理：缩放尺寸、转成 [C,H,W] 浮点张量。
            query = build_transform()(image.convert("RGB"))
        # 故意申请超过库容量的 99 个结果，验证程序会按实际 8 条记录截断。
        results = search_similar_images(collection, query, encoder, img_cnt=99)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(len(results), 8)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(results[0]["filename"], "2.jpg")
        # 测试要求下方代码必须抛出指定异常；若没有抛出，说明保护逻辑失效，测试会失败。
        with self.assertRaises(FileExistsError):
            # 调用完整建库流程，使用所选编码器、目录、批次、设备与覆盖开关。
            create_embeddings(encoder, self.data, db)
        # 测试要求下方代码必须抛出指定异常；若没有抛出，说明保护逻辑失效，测试会失败。
        with self.assertRaises(ValueError):
            # 故意传入随机新模型，其权重指纹与建库模型不同；应当拒绝误查询。
            get_embedding_collection(ConvEncoder(), db)
        # 建立 Flask 测试客户端；直接模拟 HTTP 调用，不占用真实服务器端口。
        client = create_app(dict(TESTING=True, ARTIFACT_DIR=self.artifacts, DATA_DIR=self.data)).test_client()
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(client.get("/").status_code, 200)
        # for 循环逐项遍历 ['/static/style.css', '/static/app.js']，每轮把当前元素赋给 path，再执行缩进内操作。
        for path in ["/static/style.css", "/static/app.js"]:
            # 用测试客户端读取响应，并在退出 with 时关闭响应占用的文件资源。
            with client.get(path) as static_response:
                # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
                self.assertEqual(static_response.status_code, 200)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(client.get("/health").json["device"], "cpu")
        # for 循环逐项遍历 ['/denoising', '/classification', '/simimages']，每轮把当前元素赋给 endpoint，再执行缩进内操作。
        for endpoint in ["/denoising", "/classification", "/simimages"]:
            # 对当前遍历到的业务接口发送真实格式的上传请求，取回响应供断言检查。
            response = client.post(endpoint, data=self.upload(), content_type="multipart/form-data")
            # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
            self.assertEqual(response.status_code, 200, response.data)
        # 模拟浏览器上传图片并请求检索；multipart/form-data 是带文件的表单编码格式。
        response = client.post("/simimages", data=self.upload(), content_type="multipart/form-data")
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(len(response.json["results"]), 5)
        # 用测试客户端读取响应，并在退出 with 时关闭响应占用的文件资源。
        with client.get(response.json["results"][0]["url"]) as image_response:
            # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
            self.assertEqual(image_response.status_code, 200)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(client.post("/denoising").status_code, 400)
        # 创建字典：花括号内是“键:值”映射，之后可以通过方括号按键读取。
        bad = {"image": (io.BytesIO(b"not an image"), "bad.jpg")}
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(client.post("/classification", data=bad).status_code, 400)
        # 故意指定没有模型的目录，检查首页能打开、模型接口能明确返回缺失提示。
        missing = create_app(dict(TESTING=True, ARTIFACT_DIR=self.root / "missing")).test_client()
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(missing.get("/").status_code, 200)
        # 断言实际值必须等于预期值；不相等会让测试失败，帮助定位形状、数量或 HTTP 状态错误。
        self.assertEqual(missing.post("/classification", data=self.upload()).status_code, 503)
        # 同时覆盖用户实际使用的命令行函数和独立评估流程。
        # 从 contextlib 导入 redirect_stdout，用于提供上下文管理器，例如验证时切换梯度记录方式或测试时捕获打印。
        from contextlib import redirect_stdout
        # 从 predict 导入 main as predict_main；函数/类在对应文件实现，本文件直接复用。
        from predict import main as predict_main
        # 从 scripts.evaluate 导入 main as evaluate_main；函数/类在对应文件实现，本文件直接复用。
        from scripts.evaluate import main as evaluate_main
        # for 循环逐项遍历 ['denoise', 'classify', 'similar']，每轮把当前元素赋给 task，再执行缩进内操作。
        for task in ["denoise", "classify", "similar"]:
            # 临时把 print 输出接到内存缓冲区，测试调用仍真实执行，只是不刷屏。
            with redirect_stdout(io.StringIO()):
                # 以参数列表模拟用户命令行，分别验证去噪、分类和检索入口，产物仍写入临时目录。
                predict_main([task, str(self.data / "2.jpg"), "--artifact-dir", str(self.artifacts),
                              "--output", str(self.root / "denoised.png")])
        # 断言这个条件必须为真，例如文件应已生成、固定种子的噪声应一致。
        self.assertTrue((self.root / "denoised.png").is_file())
        # for 循环逐项遍历 ['denoising', 'classification', 'similarity']，每轮把当前元素赋给 task，再执行缩进内操作。
        for task in ["denoising", "classification", "similarity"]:
            # 临时把 print 输出接到内存缓冲区，测试调用仍真实执行，只是不刷屏。
            with redirect_stdout(io.StringIO()):
                # 按已保存的临时模型和 split 文件，分别运行三个任务的独立验证集评估。
                evaluate_main([task, "--artifact-dir", str(self.artifacts), "--data-dir", str(self.data),
                               "--labels", str(self.labels)])

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 启动 unittest 测试运行器，自动寻找并执行测试类中 test_ 开头的方法。
    unittest.main()
