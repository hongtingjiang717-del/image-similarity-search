import torch
from torch import nn,optim
from torch.utils.data import DataLoader

from common.utils import *
from classification_config import *
from classification_data import create_dataset
from classification_model import  Classifier
from classification_engine import  *

from tqdm import tqdm  #进度条库

#定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(SEED)

#1.创建数据集
train_dataset,val_dataset,_ = create_dataset()
print("数据集创建完成!")

#2.创建数据加载器
train_loader = DataLoader(train_dataset,batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset,batch_size=VAL_BATCH_SIZE)
print("数据加载器创建完成! ")

#3.定义模型
classifier = Classifier() #这是一个放方法
classifier.to(device)

#4.定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(classifier.parameters(),lr=LEARNING_RATE)

#5.训练模型
#保存最小的验证损失
min_val_loss = float('inf')
for epoch in tqdm(range(EPOCHS)):
    #执行训练一个轮次
    train_loss = train_epoch(classifier,device=device,train_loader=train_loader,optimizer=optimizer,loss_fn=loss_fn)
    #执行验证
    val_loss = val_step(classifier, device=device, val_loader=val_loader, loss_fn=loss_fn)
    print(f"\nEpoch:{epoch+1}/{EPOCHS},Train_loss:{train_loss:.6f},Val_loss:{val_loss:.6f}")

    #判断如果验证损失下降，就保存模型
    if val_loss < min_val_loss:
        print("验证集损失下降，保存模型.....")
        min_val_loss = val_loss #更新参数
        #保存最小的验证损失的模型
        torch.save(classifier.state_dict(),CLASSIFIER_MODEL_NAME) #但这一句话的意思是保存全部轮次之后的模型
    else:
        print("验证损失没有减小，不保存模型！")

print("训练完成！")

'''
数据集创建完成!
数据加载器创建完成! 
  0%|          | 0/10 [00:00<?, ?it/s]
Epoch:1/10,Train_loss:0.211512,Val_loss:0.073883
验证集损失下降，保存模型.....
 20%|██        | 2/10 [01:28<05:48, 43.51s/it]
Epoch:2/10,Train_loss:0.078469,Val_loss:0.062436
验证集损失下降，保存模型.....

Epoch:3/10,Train_loss:0.058793,Val_loss:0.055588
验证集损失下降，保存模型.....
 40%|████      | 4/10 [03:05<04:42, 47.15s/it]
Epoch:4/10,Train_loss:0.050680,Val_loss:0.060253
验证损失没有减小，不保存模型！

Epoch:5/10,Train_loss:0.043331,Val_loss:0.061851
验证损失没有减小，不保存模型！
 60%|██████    | 6/10 [04:38<03:06, 46.67s/it]
Epoch:6/10,Train_loss:0.038293,Val_loss:0.046357
验证集损失下降，保存模型.....

Epoch:7/10,Train_loss:0.031500,Val_loss:0.041565
验证集损失下降，保存模型.....
 80%|████████  | 8/10 [06:08<01:32, 46.34s/it]
Epoch:8/10,Train_loss:0.029207,Val_loss:0.045376
验证损失没有减小，不保存模型！
 90%|█████████ | 9/10 [06:49<00:44, 44.51s/it]
Epoch:9/10,Train_loss:0.027536,Val_loss:0.052459
验证损失没有减小，不保存模型！

Epoch:10/10,Train_loss:0.023559,Val_loss:0.064597
验证损失没有减小，不保存模型！
训练完成！
100%|██████████| 10/10 [07:39<00:00, 45.92s/it]

第7轮之后就过拟合了我滴妈！
'''