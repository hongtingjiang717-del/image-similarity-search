# 【初学者阅读说明】
# 这是网页后端：浏览器发 HTTP 请求，Flask 路由函数处理请求并返回结果。
# GET / 返回 HTML；POST /denoising、/classification、/simimages 接收 image 文件字段。
# 图片由 PIL 解码，模型计算由 common.inference.InferenceService 完成。
# JSON 是浏览器与 Python 交换字典/列表的文本格式；jsonify 会设置正确响应类型。
# 状态码：200 成功；400 输入有问题；413 上传过大；503 所需模型/索引尚未准备好。
# create_app 是应用工厂：每调用一次就创建一套应用，便于正常启动和隔离测试。

"""本地 Flask 演示：python -m web.web_app；默认 CPU，不需要训练权重即可打开首页。"""
# 导入 argparse，解析终端中的 --epochs、--device 等命令行参数
import argparse
# 导入 base64，把图片二进制数据编码成便于传输的文本
import base64
# 从 io 导入 BytesIO，用于在内存中读写字节或文本，不必先创建硬盘文件。
from io import BytesIO
# 从 pathlib 导入 Path，用于用 Path 对象处理文件与目录路径。
from pathlib import Path
# 从 PIL 导入 Image, UnidentifiedImageError，用于通过 Pillow 打开、转换和生成图片。
from PIL import Image, UnidentifiedImageError
# 从 flask 导入 Flask, jsonify, render_template, request, send_from_directory, url_for，用于接收网页请求、注册路由、渲染模板、返回 JSON 或图片文件。
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
# 从 werkzeug.exceptions 导入 RequestEntityTooLarge，用于使用 Flask 底层 HTTP 工具提供的异常类型。
from werkzeug.exceptions import RequestEntityTooLarge
# 从 common.inference 导入 InferenceService；函数/类在对应文件实现，本文件直接复用。
from common.inference import InferenceService
# 从 common.paths 导入 ARTIFACT_DIR, DATA_DIR, DEFAULT_DEVICE；函数/类在对应文件实现，本文件直接复用。
from common.paths import ARTIFACT_DIR, DATA_DIR, DEFAULT_DEVICE

# 输入可选配置字典，返回 Flask 应用对象；测试可传入临时模型与数据目录。
def create_app(test_config=None):
    # 创建 Flask 应用；__name__ 帮助框架定位本模块旁边的 templates 和 static 目录。
    app = Flask(__name__)
    # 更新应用配置；关键字参数或字典中的键会成为 app.config 里可以查询的项目。
    app.config.update(MAX_CONTENT_LENGTH=8 * 1024 * 1024, ARTIFACT_DIR=ARTIFACT_DIR,
                      DATA_DIR=DATA_DIR, DEVICE=DEFAULT_DEVICE, DB_PATH=None)
    # 条件分支：只有 test_config 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
    if test_config:
        # 更新应用配置；关键字参数或字典中的键会成为 app.config 里可以查询的项目。
        app.config.update(test_config)
    # 创建推理服务，登记模型目录、设备和数据库路径；此时尚不加载所有权重。
    service = InferenceService(app.config["ARTIFACT_DIR"], app.config["DEVICE"], app.config["DB_PATH"])
    # 把服务登记到 Flask 的扩展字典，测试或其他代码可通过同一应用取到这份服务。
    app.extensions["inference_service"] = service

    # 从当前请求的 image 字段读取图片，检查是否可解码，返回 RGB 的 PIL 图。
    def uploaded_image():
        # 从本次 HTTP 请求的文件表单中取 image 字段；缺少字段时 get 返回 None。
        upload = request.files.get("image")
        # 条件分支：只有 upload is None or not upload.filename 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
        if upload is None or not upload.filename:
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("请选择图片，并使用 image 字段上传。")
        # try 包住可能失败的操作；只有发生匹配异常时才进入后面的 except，正常时继续后续流程。
        try:
            # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
            with Image.open(upload.stream) as image:
                # 条件分支：只有 image.width * image.height > 20000000 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
                if image.width * image.height > 20_000_000:
                    # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                    raise ValueError("图片像素过大，请缩小后上传。")
                # 把 image.convert('RGB') 交还给调用方，并结束本次函数调用；return 不会打印结果。
                return image.convert("RGB")
        # 捕获指定异常；as exc 把异常对象保存为 exc；不匹配的异常不会由这个分支处理。
        except OSError as exc:
            # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
            raise ValueError("图片格式无效或文件已损坏。") from exc

    # 将图片编码为 PNG 字节，再转 Base64 字符串，便于放进 JSON 返回给浏览器。
    def encode_image(image):
        # 建立内存中的字节缓冲区，像临时文件一样写 PNG，但不落到硬盘。
        stream = BytesIO()
        # 把 PIL 图片编码成真正的 PNG 二进制内容，写进内存缓冲区。
        image.save(stream, format="PNG")
        # 取 PNG 字节，Base64 编码为可放进 JSON 的 ASCII 文本；它不是图片 URL 本身。
        return base64.b64encode(stream.getvalue()).decode("ascii")

    # 处理图片或参数错误，返回 JSON 错误信息和 HTTP 400。
    # 下面的 @app.errorhandler 注册异常处理函数；发生对应错误时 Flask 自动调用。
    @app.errorhandler(ValueError)
    @app.errorhandler(UnidentifiedImageError)
    @app.errorhandler(Image.DecompressionBombError)
    def bad_image(exc):
        # 把多个结果打包为元组返回；调用处可以用 a,b=函数(...) 按顺序接住。
        return jsonify(error=str(exc)), 400

    # 处理模型或索引文件缺失，返回 HTTP 503，提示先完成训练/建库。
    # 下面的 @app.errorhandler 注册异常处理函数；发生对应错误时 Flask 自动调用。
    @app.errorhandler(FileNotFoundError)
    def missing_artifact(exc):
        # 把多个结果打包为元组返回；调用处可以用 a,b=函数(...) 按顺序接住。
        return jsonify(error=str(exc)), 503

    # 处理超过上传上限的请求，返回 HTTP 413。
    # 下面的 @app.errorhandler 注册异常处理函数；发生对应错误时 Flask 自动调用。
    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(exc):
        # 把多个结果打包为元组返回；调用处可以用 a,b=函数(...) 按顺序接住。
        return jsonify(error="上传文件不能超过 8 MB。"), 413

    # 浏览器访问首页时调用，渲染模板得到最终 HTML 文本。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.get("/")
    def index():
        # 读取 HTML 模板并执行其中的 Jinja 表达式，返回浏览器可以显示的 HTML。
        return render_template("index.html")

    # 返回服务能否响应和各权重文件是否存在；文件存在不等于模型效果合格。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.get("/health")
    def health():
        # 创建字典：花括号内是“键:值”映射，之后可以通过方括号按键读取。
        paths = {"denoising": "denoising/denoiser.pt", "classification": "classification/classifier.pt",
                 "similarity": "similarity/deep_encoder.pt"}
        # 把结果字典转为 JSON 响应；若同时返回整数（如 400/503），该整数指定 HTTP 状态码。
        return jsonify(status="ok", device=str(service.device),
                       weights={key: (service.artifact_dir / value).is_file() for key, value in paths.items()})

    # 按文件名从配置的数据目录返回图片文件，供浏览器展示检索结果。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.get("/dataset/<path:filename>")
    def dataset_image(filename):
        # 把 send_from_directory(app.config['DATA_DIR'], filename) 交还给调用方，并结束本次函数调用；return 不会打印结果。
        return send_from_directory(app.config["DATA_DIR"], filename)

    # 读取上传图，读取是否演示加噪的开关，调用服务并返回两张 Base64 图片。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.post("/denoising")
    def denoising():
        # 读取并验证当前请求上传的图片，得到 RGB PIL 对象，尚未进行模型计算。
        image = uploaded_image()
        # 表单值是字符串；先转小写再与 true 比较，缺省表示不额外加噪。
        add_noise = request.form.get("add_noise", "false").lower() == "true"
        # 调用公共服务，解包得到实际输入图和去噪输出图。
        noisy, output = service.denoise(image, add_noise)
        # 把结果字典转为 JSON 响应；若同时返回整数（如 400/503），该整数指定 HTTP 状态码。
        return jsonify(noisy_img=encode_image(noisy), denoised_image=encode_image(output))

    # 读取上传图片，调用分类服务，把类别结果转换成 JSON 响应。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.post("/classification")
    def classification():
        # 把结果字典转为 JSON 响应；若同时返回整数（如 400/503），该整数指定 HTTP 状态码。
        return jsonify(service.classify(uploaded_image()))

    # 检索上传图片，为每个结果增加可访问的图片 URL，再返回 JSON。
    # 下面的 @app 路由装饰器把 URL 与这个 Python 函数关联起来，收到请求时由 Flask 调用。
    @app.post("/simimages")
    def simimages():
        # 读取上传图并查询最多 5 个相似商品；结果中的距离不是分类置信度。
        results = service.similar(uploaded_image(), 5)
        # for 循环逐项遍历 results，每轮把当前元素赋给 item，再执行缩进内操作。
        for item in results:
            # 用路由函数名生成图片地址，包含真实文件名，避免前端把记录序号误当文件名。
            item["url"] = url_for("dataset_image", filename=item["filename"])
        # 把结果字典转为 JSON 响应；若同时返回整数（如 400/503），该整数指定 HTTP 状态码。
        return jsonify(results=results, indices_list=[item["id"] for item in results])

    # 把配置好路由和错误处理器的应用对象交回调用者。
    return app

# 提供一个可供 Flask/其他启动器导入的应用；模型权重仍按需延迟加载。
app = create_app()

# 命令行入口函数；argv=None 表示默认读取终端参数，测试时可传入字符串列表。
def main(argv=None):
    # 创建命令行参数解析器；description 指定运行 --help 时展示的说明。
    parser = argparse.ArgumentParser(description=__doc__)
    # 登记命令行参数 --host；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--host", default="127.0.0.1")
    # 登记命令行参数 --port；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=int 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--port", type=int, default=9000)
    # 登记命令行参数 --device；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 未提供该选项时采用 default 的值。
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    # 登记命令行参数 --artifact-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    # 登记命令行参数 --data-dir；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。 未提供该选项时采用 default 的值。
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    # 登记命令行参数 --db-path；以 -- 开头的是可选参数，否则是必须按顺序提供的位置参数。 type=Path 会把终端字符串转换成指定类型。
    parser.add_argument("--db-path", type=Path)
    # 解析终端参数到 args 对象；例如 --batch-size 4 会保存为 args.batch_size=4。
    args = parser.parse_args(argv)
    # 把终端参数组织成配置字典，创建使用这些目录和设备的新应用，随后启动它。
    application = create_app(dict(DEVICE=args.device, ARTIFACT_DIR=args.artifact_dir,
                                  DATA_DIR=args.data_dir, DB_PATH=args.db_path))
    # 在指定地址和端口启动本地演示服务；debug=False 不启用开发调试器。
    application.run(host=args.host, port=args.port, debug=False)

# 只有把本文件作为程序入口运行时才执行下面代码；被 import 导入时不会自动启动训练/服务。
if __name__ == "__main__":
    # 调用本文件的命令行入口；真正开始执行前面定义的流程。
    main()
