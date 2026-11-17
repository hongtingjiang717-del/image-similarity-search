import torch
from torch import nn,optim
from torch.utils.data import DataLoader

from common.utils import *
from denoising_config import *
from denoising_data import create_dataset
from denoising_model import  ConvDenoiser
from denoising_engine import  *

from tqdm import tqdm  #进度条库

#定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(SEED)

#1.创建数据集
train_dataset,val_dataset,_ = create_dataset()
print("数据集创建完成！")

#2.创建数据加载器
train_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset,batch_size=VAL_BATCH_SIZE,shuffle=False,drop_last=True)
print("数据加载器创建完成！ ")

#3.定义模型
denoiser = ConvDenoiser() #这是一个放方法
denoiser.to(device)

#4.定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.Adam(denoiser.parameters(),lr=LEARNING_RATE)

#5.训练模型
#保存最小的验证损失
min_val_loss = float('inf')
for epoch in tqdm(range(EPOCHS)):
    #执行训练一个轮次
    train_loss = train_epoch(denoiser,device=device,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn)
    #执行验证
    val_loss = val_step(denoiser, device=device, val_loader=val_loader, loss_fn=loss_fn)
    print(f"\nEpoch:{epoch+1}/{EPOCHS},Train_loss:{train_loss:.6f},Val_loss:{val_loss:.6f}")

    #判断如果验证损失下降，就保存模型
    if val_loss < min_val_loss:
        print("验证集损失下降，保存模型.....")
        min_val_loss = val_loss #更新参数
        #保存最小的验证损失的模型
        torch.save(denoiser.state_dict(),DENOISER_MODEL_NAME) #但这一句话的意思是保存全部轮次之后的模型
    else:
        print("验证损失没有减小，不保存模型！")

print("训练完成！")


#训练30epoch的结果
'''
数据集创建完成！
数据加载器创建完成！ 
  3%|▎         | 1/30 [03:41<1:47:09, 221.72s/it]
Epoch:1/30,Train_loss:0.038139,Val_loss:0.019433
验证集损失下降，保存模型.....


Epoch:30/30,Train_loss:0.008137,Val_loss:0.008150
验证集损失下降，保存模型.....
训练完成！
100%|██████████| 30/30 [1:21:26<00:00, 162.87s/it]

这个不是很好训练，训练损失和验证损失都在下降，所以没有过拟合，如果有条件可以将训练轮数继续增加
'''