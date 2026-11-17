#=================基本引擎的实现======================================================

import torch
#训练一个轮次,返回平均损失
def train_epoch(model, device, train_loader, loss_fn,optimizer):
    model.train()  #开启训练模式
    train_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        #前向传播
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        #更新参数
        optimizer.step()
        #梯度清零
        optimizer.zero_grad()
        #累加损失
        train_loss += loss.item()
    #计算本轮平均损失,len(train_loader)批次数量
    return train_loss / len(train_loader)

#验证一个轮次,返回平均验证损失
def val_step(model, device, val_loader, loss_fn):
    model.eval()  #开启验证模式
    val_loss = 0.0
    val_dataset_num = 0 #累积验证集数据总量
    #禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            #前向传播
            output = model(input)
            loss = loss_fn(output, target)
            #累加损失
            # val_loss += loss.item()
            val_loss += loss.item()*input.shape[0]
            val_dataset_num += input.shape[0]
    #计算本轮平均损失,len(train_loader)批次数量
    return val_loss / val_dataset_num

#测试一个轮次,返回评价指标-准确率
def test_step(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    test_correct_num = 0
    test_dataset_num = 0
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.to(device), target.to(device)
            #前向传播
            output = model(input)
            # 转换成预测分类
            pred = output.argmax(dim=1).cpu()
            #累积预计准确数量以及总数量
            test_correct_num += pred.eq(target).sum().item()
            test_dataset_num += input.shape[0]

    #计算本轮平均损失,len(train_loader)批次数量
    return test_correct_num / test_dataset_num