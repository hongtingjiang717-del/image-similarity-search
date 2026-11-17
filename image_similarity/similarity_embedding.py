import os

import chromadb
import torchvision
from PIL import Image
from chromadb.api.types import Images, EmbeddingFunction, Embeddings
import numpy as np
import torch
from tqdm import tqdm

from common.utils import sorted_alphanumeric
from image_similarity.similarity_config import *
from image_similarity.similarity_model import *
# from image_similarity.similarity_train import * 导入这个模块会先执行这个模块，所以刚才没注释的时候一直在训练
import torchvision.transforms as T
from math import ceil


#自定义嵌入函数
#针对图片自编码器，定义嵌入函数
class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    def __init__(self, model):
        self.model = model.cpu()
        return
    #call方法
    def __call__(self, input: Images) -> Embeddings:
        #将输入数据转成tensor
        input_tensor = torch.tensor(np.array(input))
        #前向传播
        with torch.no_grad():
            output = self.model(input_tensor)
        return output.numpy()
# 准备工作：
# 获取所有图片id和图像数据的对应关系(最后获得一个字典)
def get_id2image(main_dir,transform):
    #定义字典
    id2image = {}
    # 先获取所有图片文件名的列表
    img_names = sorted_alphanumeric(os.listdir(main_dir))
    #由于这个过程非常的慢，所以我们要加一个进度条
    with tqdm(total=len(img_names)) as pbar:
        #遍历文件名列表，索引号就是id，打开图片记录数据
        for i,image_names in enumerate(img_names):
            #1.根据索引号找到文件名，读取图片数据
            img_path = os.path.join(main_dir,image_names)
            image = Image.open(img_path).convert("RGB")
            #2.将原始图片转换为符合模型输入要求的张量
            image_tensor = transform(image)
            #3.转成ndarray写入字典
            id2image[str(i)] = image_tensor.numpy()
            pbar.update(1)  #每加载一张，进度加1，注意！不要写到for循环之外
    return id2image


#准备工作:获取集合
def get_collection(encoder):
    # 1.创建客户端
    client = chromadb.PersistentClient(
        os.path.join(
            '..',
            PACKAGE_NAME,
            CHROAM_BACKEND_PATH
        )
    )
    #2.获取或者创建集合，指定嵌入函数
    collection = client.get_or_create_collection(
        name = "image_similarity",
        embedding_function = ImageEmbeddingFunction(encoder)
    )
    return collection

#将所有的图片写入向量数据库(upsert)，自动生成嵌入向量的表达
def create_embeddings(encoder):
    #1.获取图像的id和图像数据
    transform = T.Compose([
        T.Resize((IMG_HEIGHT, IMG_WIDTH)),
        T.ToTensor(),
    ])
    print("正在加载图片...")
    id2image = get_id2image(IMG_PATH, transform)

    print("图片加载完成！")
    ids = list(id2image.keys())
    images = list(id2image.values())

    # 2.获取集合
    collection = get_collection(encoder)

    #3.分别写入向量数据库，自动生成嵌入向量
    print("正在写入向量数据库...")
    for i in range( ceil(len(ids) / CHROMA_INSERT_BATCH_SIZE)):
        start = i * CHROMA_INSERT_BATCH_SIZE
        end =(i+1) *CHROMA_INSERT_BATCH_SIZE
        collection.upsert(
            ids = ids[start:min(end,len(ids))],
            images = images[start:min(end,len(ids))],
        )
    print("向量写入完成!")

#对输入的图像进行相似度检索，返回最相似的5个图片id
def search_similar_image_ids(collection, image_tensor,cnt):
    result = collection.query(
        query_images = [image_tensor.numpy()],
        n_results=cnt
    )
    ids = [int(id) for id in result["ids"][0]]
    return ids

#测试程序
if __name__ == "__main__":
    # 1.加载编码器模型
    encoder = ConvEncoder()
    state_dict = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(state_dict)
    print("模型加载完毕！")
    # 2.生成嵌入向量
    create_embeddings(encoder)