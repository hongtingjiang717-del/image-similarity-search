# 【初学者阅读说明】
# 分类样本是 (image, target)：image 是图片张量，target 是 0～4 的整数类别。
# CSV 的 id 必须与图片文件名对应：例如 15.jpg 查 id=15，不能按它在列表中的位置去查。
# 例如文件只有 2.jpg、8.jpg 时，dataset[1] 是 8.jpg，要查 id=8，不是 id=1。

"""分类标签通过图片文件名中的 id 查找，不能使用数据集索引代替 id。"""
# 导入 csv，按 CSV 的行与列读取/写入分类标签
import csv
# 从 common.data 导入 ImageFolderDataset, sorted_alphanumeric；函数/类在对应文件实现，本文件直接复用。
from common.data import ImageFolderDataset, sorted_alphanumeric
# 包内相对导入：前面的点表示当前包；从 classification_config 取出 classification_names，避免误找同名顶层模块。
from .classification_config import classification_names

# 指定 from 模块 import * 时公开哪些名字；不会自动创建这些类或函数。
__all__ = ["ImageDataset"]

# 定义 ImageDataset 数据集类并继承父类能力；只重写与当前任务不同的部分。
class ImageDataset(ImageFolderDataset):
    # 创建对象时自动执行的初始化方法；self 表示正在创建的这个对象。
    # self.xxx 把数据保存在对象中，让这个对象的其他方法以后也能使用。
    def __init__(self, main_dir, fashion_label_path, transform=None):
        # 调用父类的数据集初始化，复用图片枚举和预处理逻辑，避免重复实现。
        super().__init__(main_dir, transform)
        # 建立空字典，用图片 id 当键、类别整数当值，之后按 id 直接查询。
        self.label_dict = {}
        # with ... as ... 管理文件或图片资源：进入时打开，块结束后自动关闭，即使中途发生异常也尽量释放资源。
        with open(fashion_label_path, encoding="utf-8-sig", newline="") as stream:
            # 把 CSV 每一行读成字典，例如 {"id":"15", "target":"2"}；读出的字段值先是字符串。
            rows = csv.DictReader(stream)
            # 条件分支：只有 not {'id', 'target'}.issubset(rows.fieldnames or []) 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
            if not {"id", "target"}.issubset(rows.fieldnames or []):
                # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                raise ValueError("标签 CSV 必须包含 id,target 列")
            # for 循环逐项遍历 rows，每轮把当前元素赋给 row，再执行缩进内操作。
            for row in rows:
                # 通过列名取字段，并从字符串转成整数；解包到图片编号与类别两个变量。
                image_id, target = int(row["id"]), int(row["target"])
                # 条件分支：只有 image_id in self.label_dict 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
                if image_id in self.label_dict:
                    # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                    raise ValueError(f"标签 CSV 存在重复 id：{image_id}")
                # 条件分支：只有 target not in classification_names 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
                if target not in classification_names:
                    # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                    raise ValueError(f"未知分类标签：{target}")
                # 用方括号给字典增加一项映射，例如标签表中 id=15 的类别。
                self.label_dict[image_id] = target
        # 建立与图片列表一一对应的标签列表，后面按数据集索引即可取标签。
        self.targets = []
        # for 循环逐项遍历 self.paths，每轮把当前元素赋给 path，再执行缩进内操作。
        for path in self.paths:
            # 条件分支：只有 not path.stem.isdecimal() or int(path.stem) not in self.label_dict 为真时才执行下方缩进代码；否则跳过或进入对应 else 分支。
            if not path.stem.isdecimal() or int(path.stem) not in self.label_dict:
                # 主动抛出有说明的异常，停止当前错误路径；上层命令行或 Flask 错误处理器决定如何向用户展示。
                raise ValueError(f"图片没有对应的数字 id 标签：{path.name}")
            # path.stem 是不含后缀的文件名，例如 15.jpg→15；转换为整数查字典，再追加对应类别。
            self.targets.append(self.label_dict[int(path.stem)])

    # Python 执行 dataset[idx] 时自动调用这个方法；DataLoader 也通过它逐个取样本。
    def __getitem__(self, idx):
        # 同时返回图片张量和正确整数标签；DataLoader 会把多个整数自动组成一维标签张量。
        return self.load_image(idx), self.targets[idx]
