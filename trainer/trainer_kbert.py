########################################################################
# 训练、验证和测试函数
########################################################################
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
import time
from torch.utils.data import TensorDataset, DataLoader
from dataprocess.dataprocess_kbert import Dataprocess 
from trainer.focal_loss import FocalLoss
def get_metrics(true_res, pred_res):
    acc = accuracy_score(y_true=true_res, y_pred=pred_res)
    pre = precision_score(y_true=true_res, y_pred=pred_res, average="macro")
    rec = recall_score(y_true=true_res, y_pred=pred_res, average="macro")
    f1 = f1_score(y_true=true_res, y_pred=pred_res, average="macro")
    return acc, pre, rec, f1


# 模型验证和测试
def evaluate(args, model, data_loader):
    print("Evaluation Start======")
    model.eval()
    true_res, pred_res = [], []
    with torch.no_grad():  # 计算的结构在计算图中,可以进行梯度反转等操作
        for data in data_loader:
            input_ids, attention_masks, token_type_ids, labels = tuple(t.to(args.device) for t in data)  # 将三个tuple元素传到服务器中            
            y_pred = model(input_ids, attention_masks, token_type_ids) 
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()  # 将概率矩阵转换成标签并变成list类型
            pred_res.extend(y_pred)  # 将标签值放入列表
            true_res.extend(labels.cpu().numpy().tolist()) # 将真实标签转换成list放在列表中
    data_acc, data_pre, data_rec, data_f1 = get_metrics(true_res, pred_res)
    return data_acc, data_pre, data_rec, data_f1

# 模型训练
def train(args, model, train_loader, dev_loader, optimizer):
    best_acc = 0.0
    criterion = FocalLoss()
    #criterion = nn.CrossEntropyLoss()    
    for epoch in range(args.num_epochs):
        start = time.time()
        steps = 0     # 用来打印后面的输出
        model.train()
        print("***************training epoch{}************".format(epoch + 1))
        running_loss = 0.0
        for batch in train_loader:
            input_ids, attention_masks, token_type_ids, labels = tuple(t.to(args.device) for t in batch)  # 将三个tuple元素传到服务器中
             
            # 1、前向传播
            y_pred = model(input_ids, attention_masks, token_type_ids)              
            loss = criterion(y_pred, labels)   #输入的y_pred=[batch数，类别数], label=[类别数]

            # 2、反向传播
            optimizer.zero_grad()
            loss.backward()
            # 3、梯度更新
            optimizer.step()
            running_loss += loss.item()
           # 只打印五次结果            
            steps = steps+1
            if steps % 5 == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(epoch + 1, steps, len(train_loader), running_loss / steps, time.time() - start))
        # 一轮训练结束，在验证集测试
        model.eval()
        valid_acc, valid_pre, valid_rec, valid_f1 = evaluate(args, model, dev_loader)
        if valid_acc > best_acc:
            best_acc = valid_acc
            #torch.save(model.state_dict(), "./save_models/best_model.pkl")  # 保存最好的模型
        print("current acc is {:.4f},best acc is {:.4f}".format(valid_acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))

# 数据批量处理，
def dataLoader(args, state, index):
    list_all = Dataprocess(args, state, index)
       
    input_ids = []
    attention_masks = []
    token_type_ids = []    # 如果是输入两个句子，则需要用这个参数   
    labels = []
    for i, dict_line in enumerate(list_all):
        input_id, attention_mask, token_type_id, label = dict_line.get('input_ids'), dict_line.get('attention_masks'), dict_line.get("token_type_ids"), dict_line.get("label_ids")    # label 是一个包含1861个int的list
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)
        attention_masks.append(attention_mask)        
        labels.append(label)
    
    #torch.Tensor()默认转成tensor.float32，如果要tensor.int64，需要LongTensor
    input_ids = torch.LongTensor(input_ids)      # tensor[1861,70]
    token_type_ids = torch.LongTensor(token_type_ids)   # tensor[1861,70]
    attention_masks = torch.LongTensor(attention_masks)    # tensor[1861,70]
    labels = torch.LongTensor(labels)    # tensor[1861,1]
    

    datas = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    loader = DataLoader(dataset=datas,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)
    return loader