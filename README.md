# 智图寻宝完整项目

本项目根据《尚硅谷大模型技术之智图寻宝》V2.0.0 的正式代码清单整理，包含图像去噪、五类商品分类、相似图片检索和 Flask 演示页面。保留教材网络结构，修复运行错误，默认使用 CPU。原数据、学习笔记、Notebook 和旧权重文件均保留。

## GitHub 版本说明

仓库上传源码、测试、依赖清单和项目说明，不包含本机商品图片、模型权重、向量数据库、源码备份或本地环境。下文的 24,853 张图片统计来自本机数据检查，不表示 GitHub 仓库已附带这些图片。旧权重仍保留在本机，不随仓库上传。

克隆后可先安装依赖并运行 `python -m unittest tests.test_project -v`：测试会自动生成临时小数据，无需下载商品数据或预训练权重。网页首页也可以在没有权重时打开。

如需正式训练，请自行准备有使用权限的商品图片，放入 `common/dataset/`，确保图片文件名的数字 ID 与 `common/fashion-labels.csv` 中的 ID 对应；也可通过下文的路径参数使用自己的数据和标签。检查数据后依次训练模型、构建索引，再运行网页推理。当前项目属于课程实践与工程完善，未报告正式全量训练准确率或生产性能。

目前交付的是可训练、可推理的完整源码。没有进行全量正式训练，`artifacts/` 中也没有伪造或放入测试模型；首页可以直接打开，模型接口在缺少权重时会返回明确提示。

源码已补充面向初学者的中文注释，覆盖 Python、HTML、CSS、JavaScript，以及两个历史 Notebook 的代码单元。建议从 [初学者阅读指南](docs/初学者阅读指南.md) 开始，按指南顺序阅读各文件。历史 Notebook 的原有实验行为与输出保留，注释中已标明应对照正式实现理解的旧写法。

## 项目结构

```text
Image_similarity_main/
├── common/
│   ├── dataset/                     # 已有 24,853 张商品图片
│   ├── fashion-labels.csv           # id,target 标签
│   ├── paths.py                     # 项目路径和环境变量
│   ├── data.py                      # 图片枚举、自然排序、预处理
│   ├── utils.py                     # 随机种子、设备、权重读写
│   ├── engine.py                    # 训练/验证核心循环
│   ├── training.py                  # CLI、划分、最佳权重和日志
│   └── inference.py                 # 三种模型的公共推理服务
├── image_denoising/
│   ├── denoising_config.py
│   ├── denoising_data.py
│   ├── denoising_model.py
│   ├── denoising_engine.py
│   └── denoising_train.py
├── image_classification/
│   ├── classification_config.py
│   ├── classification_data.py
│   ├── classification_model.py
│   ├── classification_engine.py
│   └── classification_train.py
├── image_similarity/
│   ├── similarity_config.py
│   ├── similarity_data.py
│   ├── similarity_model.py
│   ├── similarity_engine.py
│   ├── similarity_train.py
│   └── similarity_embeddings.py     # Chroma 建库和查询
├── web/
│   ├── web_app.py
│   ├── templates/index.html
│   └── static/{style.css,app.js}
├── scripts/{check_dataset.py,evaluate.py}
├── tests/test_project.py            # 临时数据上的完整集成验证
├── artifacts/                      # 正式训练与建库后自动生成
├── backups/                        # 本次整理前的 Python 源码备份
├── docs/                           # 修正说明、源码对应关系、验证记录
├── predict.py                      # 单张图片命令行推理
├── requirements.txt
└── requirements-cpu.txt
```

五类标签：`0 上身衣服`、`1 鞋`、`2 包`、`3 下身衣服`、`4 手表`。

## 环境与启动目录

本次实际使用的解释器：

```text
D:\Anaconda_envs\envs\image_similarity_main\python.exe
```

打开 PowerShell，激活你已有的 CPU 环境，并进入项目根目录：

```powershell
conda activate image_similarity_main
Set-Location 'D:\desktop\尚硅谷\Image_similarity_main'
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

本次未安装、升级或修改你的环境。已验证的版本见 `docs/环境版本.json`。

如在新机器创建 CPU 环境：

```powershell
conda create -n image_similarity_main python=3.12 pip
conda activate image_similarity_main
python -m pip install -r requirements-cpu.txt
```

已有兼容 torch/torchvision 时，按需安装 `requirements.txt` 中缺少的依赖即可。torch 和 torchvision 需要使用相互兼容的版本，CPU 与 CUDA 安装方式不要混用。

所有训练与 Web 入口采用模块方式 `python -m 包名.模块名`；不要在包目录里直接运行 `python xxx_train.py`。

## 先检查项目

```powershell
python -m scripts.check_dataset
python -m scripts.check_dataset --verify-images
python -m unittest tests.test_project -v
```

第二条会解码检查全部图片。第三条只用临时生成的 8 张图片，在临时目录完成三种模型各一轮训练、建库与 HTTP 请求，不会覆盖正式权重，也不能用于判断模型效果。

## 训练三个模型

```powershell
python -m image_denoising.denoising_train --device cpu --preview
python -m image_classification.classification_train --device cpu
python -m image_similarity.similarity_train --device cpu
```

默认 30 轮、批次 32、学习率 0.001。配置位于各模块的 `*_config.py`。`--help` 可查看参数。去噪使用 Adam；分类与检索重建使用 AdamW。输入均为 RGB 64×64。

先做少量真实图片的流程检查，可使用独立输出目录：

```powershell
python -m image_denoising.denoising_train --epochs 1 --limit 16 --batch-size 4 --output-dir artifacts/smoke-denoising --preview
python -m image_classification.classification_train --epochs 1 --limit 16 --batch-size 4 --output-dir artifacts/smoke-classification
python -m image_similarity.similarity_train --epochs 1 --limit 16 --batch-size 4 --output-dir artifacts/smoke-similarity
```

`--limit` 只用于验证流程。正式训练不要携带该参数。已有权重的输出目录会拒绝覆盖，再次实验请使用新的 `--output-dir`。当前版本支持保存最佳模型用于推理，不提供中断后恢复优化器状态的断点续训。

训练使用固定种子随机划分 75% 训练、25% 验证，保留尾批。去噪验证集固定每张图的噪声。损失按样本数量加权汇总。

默认产物：

```text
artifacts/
├── denoising/
│   ├── denoiser.pt
│   ├── history.json
│   ├── training_config.json
│   ├── split.json                  # 保存训练和验证文件名
│   └── preview.png                 # 传入 --preview 时生成
├── classification/
│   ├── classifier.pt
│   ├── history.json
│   └── training_config.json
├── similarity/
│   ├── deep_encoder.pt
│   ├── deep_decoder.pt
│   ├── embeddings.npy              # 最优编码器生成的原始 [N,512] 向量
│   ├── embedding_filenames.json    # 每行向量对应的文件名
│   ├── history.json
│   └── training_config.json
└── chroma_backend/                 # 单独建库后生成
```

`embeddings.npy` 是便于检查和离线使用的导出文件；网页检索使用 Chroma，仍需执行下面的建库命令。`--limit` 训练时导出文件也只包含被抽取的图片。

## 构建图片检索库

完成相似模型正式训练后执行：

```powershell
python -m image_similarity.similarity_embeddings --device cpu
```

该步骤使用最优编码器分批处理所有图片，将 L2 归一化后的 512 维向量、真实文件名存入持久化 Chroma，使用余弦距离检索。不会下载在线嵌入模型，也不会一次加载所有图片到内存。

换了编码器权重或图片集合后，需要重新建库：

```powershell
python -m image_similarity.similarity_embeddings --device cpu --rebuild
```

`--rebuild` 仅替换 `--db-path` 指定数据库中的 `image_similarity` 集合。未指定时使用 `artifacts/chroma_backend`。如需保留旧实验，请为新实验指定独立 `--db-path`。建库与查询会校验编码器指纹；建库中断留下的集合也不会被当作完整索引使用。

## 验证模型与单张推理

按训练时保存的 `split.json` 文件名列表评估验证集；请保留原始验证图片和标签。训练图片与验证图片的列表均会保存，其他两个模型目录也有同样的 `split.json`：

```powershell
python -m scripts.evaluate denoising
python -m scripts.evaluate classification
python -m scripts.evaluate similarity
```

分类报告验证损失与准确率，另外两个模型报告重建损失。相似模型的重建损失不能代替检索质量评估，检索效果需额外设计人工相关性或带检索标签的评估集。

```powershell
python predict.py denoise Apple.jpg --output artifacts/results/apple-denoised.png
python predict.py denoise Apple.jpg --add-noise --output artifacts/results/apple-demo.png
python predict.py classify common/dataset/0.jpg
python predict.py similar common/dataset/0.jpg --count 5
```

这些命令需要先有对应训练权重，相似检索还需要已构建的 Chroma 集合。默认去噪直接处理上传图像；`--add-noise` 才额外添加噪声以复现教材演示。

## 打开网页

```powershell
python -m web.web_app
```

浏览器访问 `http://127.0.0.1:9000`。首页可以在训练前查看；接口缺少模型时返回 503 和提示，不会用随机参数给出伪结果。

选择图片后可去噪、分类和检索。默认勾选“演示加噪后去噪”；真实带噪图片可取消勾选。分类与检索可直接使用原图；去噪成功后则使用去噪图。模型置信度仅表示模型输出，不等同于已校准的正确概率。

接口：`GET /health`、`POST /denoising`、`POST /classification`、`POST /simimages`。POST 上传字段为 `image`；去噪另支持表单字段 `add_noise=true`。检索响应包含真实文件名和可直接访问的图片 URL。

## 以后迁移到其他机器

项目没有硬编码当前 Windows 盘符。保持项目、数据、模型目录结构即可迁移；可用以下环境变量指定外部路径：

| 变量 | 默认位置 |
|---|---|
| `IMAGE_SIM_DATA_DIR` | `common/dataset` |
| `IMAGE_SIM_LABELS_PATH` | `common/fashion-labels.csv` |
| `IMAGE_SIM_ARTIFACT_DIR` | `artifacts` |
| `IMAGE_SIM_DEVICE` | `cpu` |

也可通过各命令的 `--data-dir`、`--output-dir`、`--artifact-dir`、`--db-path`、`--device` 覆盖。以后在配置好 CUDA 版 PyTorch 的环境中传 `--device cuda` 即可选择 GPU；当前没有配置或租用服务器。

Windows 的 CPU 依赖版本记录不应直接当作 Linux CUDA 环境锁定文件使用。迁移后应在目标环境重新运行集成测试；当前 Flask 命令用于本地演示，没有进行生产部署配置。

## 原有文件与修正记录

- 详细问题、修复与行为调整：`docs/代码修正说明.md`。
- Word 章节与源码对应关系：`docs/文档代码对应表.md`。
- 验证范围和限制：`docs/验证记录.md`。
- 整理前 Python 源码备份：`backups/source_before_assembly_*.zip`。
- 根目录原有 `conv_autoencoder.pth` 来源/结构没有确认，不会自动冒充本项目三个模型的权重；原文件未修改。
- 原有 `study.md`、Notebook 作为学习资料保留，部分旧路径或旧代码不属于新的运行入口。
