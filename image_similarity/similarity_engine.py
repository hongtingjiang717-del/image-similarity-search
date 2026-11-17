import torch
#训练一个轮次,返回平均损失
def train_epoch(encoder,decoder, device, train_loader, loss_fn,optimizer):
    # model.train()  #开启训练模式
    encoder.train()
    decoder.train()
    train_loss = 0.0
    for input, target in train_loader:
        input, target = input.to(device), target.to(device)
        #前向传播
        vectors = encoder(input)
        output = decoder(vectors)  #自编码器的解码器部分在训练过程中被使用，用来计算loss
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
def val_step(encoder,decoder, device, val_loader, loss_fn):
    # model.eval()  #开启验证模式
    encoder.train()
    decoder.train()
    val_loss = 0.0
    #禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for input, target in val_loader:
            input, target = input.to(device), target.to(device)
            #前向传播
            vectors = encoder(input)
            output = decoder(vectors)
            loss = loss_fn(output, target)
            #累加损失
            val_loss += loss.item()
    #计算本轮平均损失,len(train_loader)批次数量
    return val_loss / len(val_loader)

#测试一个轮次,返回评价指标-平均测试误差
def test_step(encoder,decoder, device, test_loader, loss_fn):
    # model.eval()
    encoder.train()
    decoder.train()
    test_loss = 0.0
    with torch.no_grad():
        for input, target in test_loader:  #input和target是一个相同的张量
            input, target = input.to(device), target.to(device)
            #前向传播
            vectors = encoder(input)
            output = decoder(vectors)
            loss = loss_fn(output, target)
            #累加损失
            test_loss += loss.item()
    #计算本轮平均损失,len(train_loader)批次数量
    return test_loss / len(test_loader)