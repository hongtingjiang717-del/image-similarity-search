import torch
import matplotlib.pyplot as plt
from PIL import Image
from similarity_embedding import *

from image_similarity.similarity_embedding import get_collection
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder
from similarity_embedding import search_similar_image_ids
import os

if __name__ == "__main__":
    #获取数据集
    _,_,test_data = create_dataset()

    #1.获取一个测试数据
    image,_ = test_data[211]
    print(image.shape)

    #2.加载训练好的模型（编码器）
    print("正在加载模型...")
    encoder = ConvEncoder()
    state_dict = torch.load(ENCODER_MODEL_NAME)
    encoder.load_state_dict(state_dict)
    print("模型加载完毕！")

    #3.获取Chroma中的嵌入向量集合
    collection = get_collection(encoder)
    # print(collection.peek(limit=1))  #一个长度为512的向量

    #4.调用API，得到相似图片索引
    similar_image_ids = search_similar_image_ids(collection,image_tensor= image,cnt=5)
    print(similar_image_ids)

    #5.可视化的展示
    fig,axes = plt.subplots(nrows=2,ncols=5,figsize=(25,4))
    #输入测试图片
    image = image.permute(1,2,0)
    # print(image.shape)
    axes[0,2].imshow(image)

    #相似图片
    for i in range(len(similar_image_ids)):
        #根据id得到图片名称
        image_name = str(similar_image_ids[i]) + ".jpg"
        #拼接文件路径
        image_path = os.path.join(IMG_PATH,image_name)
        #读取图片
        image = Image.open(image_path).convert('RGB')
        #画图
        axes[1,i].imshow(image)
    for ax in axes.flat:
        ax.axis('off')

    plt.show()
    #向量检索比逐像素检索的优势大很多
