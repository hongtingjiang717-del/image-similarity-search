import torch
from torch import nn,optim
from torch.utils.data import DataLoader

from common.utils import *
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import  *
from similarity_engine import  *

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
print("数据加载器创建完成！")

#3.定义模型
encoder = ConvEncoder()
decoder = ConvDecoder()
encoder.to(device)
decoder.to(device)

#4.定义损失函数和优化器
loss_fn = nn.MSELoss()
#将编码器和解码器两者的参数合并一起
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params,lr=LEARNING_RATE)

#5.训练模型
#保存最小的验证损失
min_val_loss = float('inf')
for epoch in tqdm(range(EPOCHS)):
    #执行训练一个轮次
    train_loss = train_epoch(encoder,decoder,device=device,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn)
    #执行验证
    val_loss = val_step(encoder,decoder,device=device, val_loader=val_loader, loss_fn=loss_fn)
    print(f"\nEpoch:{epoch+1}/{EPOCHS},Train_loss:{train_loss:.6f},Val_loss:{val_loss:.6f}")

    #判断如果验证损失下降，就保存模型
    if val_loss < min_val_loss:
        print("验证集损失下降，保存模型.....")
        min_val_loss = val_loss #更新参数
        #保存最小的验证损失的模型
        torch.save(encoder.state_dict(),ENCODER_MODEL_NAME) #但这一句话的意思是保存全部轮次之后的模型
        torch.save(decoder.state_dict(),DECODER_MODEL_NAME)
    else:
        print("验证损失没有减小，不保存模型！")

print("训练完成！")