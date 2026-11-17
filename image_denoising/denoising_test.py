import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from common.utils import seed_everything
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import ConvDenoiser
from denoising_engine import test_step


#函数，测试一批图像的重构图像
def test_batch(model, test_loader, device):
    model.to(device)
    model.eval()
    #1.选取一批测试数据
    test_iter = iter(test_loader)
    noisy_images,images = next(test_iter) #next表示从一个迭代器里面取出下一个元素
    #2.推理预测
    with torch.no_grad():
        noisy_images = noisy_images.to(device)
        #前向传播
        outputs = model(noisy_images)
    print("输出重构图像形状：",outputs.shape)
    #3.数据转换，为画图做准备
    noisy_images = noisy_images.permute(0, 2, 3, 1).cpu().numpy()
    outputs = outputs.permute(0, 2, 3, 1).cpu().numpy()
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    #4.画图
    fig, axes = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, figsize=(25,4))  # 创建 2 行 10 列的子图
    # 第一行显示噪声图像，第二行显示重建图像
    # zip 将多个序列按位置配对；for 同时解包每一组，任一序列结束就停止。
    for imgs, row in zip([noisy_images, outputs, images], axes):  # 遍历噪声图像和重建图像
        # zip 将多个序列按位置配对；for 同时解包每一组，任一序列结束就停止。
        for img, ax in zip(imgs, row):  # 遍历每张图像和对应的子图
            # 把 [H,W,C] 数组绘制在子图中；不要直接把 [C,H,W] 的 PyTorch 排列交给它。
            ax.imshow(img)  # 显示图像，并去除多余的维度
            ax.set_axis_off()
    plt.show()  # 显示图像

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    # 1.创建数据集
    _, _, test_dataset = create_dataset()
    print("数据集创建完成")

    # 2.创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, drop_last=True)
    print("数据加载器创建完成 ")

    # 3.定义模型并加载最优参数
    denoiser = ConvDenoiser()  # 这是一个放方法
    state_dict = torch.load(DENOISER_MODEL_NAME,map_location=device)
    denoiser.load_state_dict(state_dict)
    print("模型加载完成！")

    #4.测试
    print("测试结果如下：")
    test_batch(denoiser, test_loader, device)
    test_loss = test_step(denoiser,device=device,test_loader=test_loader,loss_fn=torch.nn.L1Loss())
    print("测试集均方误差：",test_loss)