# 将所有的ecg记录和标注存为txt文件

import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.utils.data as Data
from torch.autograd import Variable
from Net import *

# 文件路径
ROOT = "E:/deeplearning/apnea-ecg-database-1.0.0/"
# 处理后文件存放路径
SEGMENTS_BASE_PATH = "E:/deeplearning/apnea-ecg-processed/"

# 训练集大小
SEGMENTS_NUMBER_TRAIN = 17045
# 测试集大小
SEGMENTS_NUMBER_TEST = 17268
# 训练集文件名称列表
TRAIN_FILENAME = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "b01", "b02", "b03", "b04", "b05",
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
]

# 训练集每个文件长度（对应多少个60s ecg信号）
TRAIN_LABEL_AMOUNT = [489, 528, 519, 492, 454,
                      510, 511, 501, 495, 517,
                      466, 577, 495, 509, 510,
                      482, 485, 489, 502, 510,
                      487, 517, 441, 429, 433,
                      484, 502, 454, 482, 466,
                      468, 429, 513, 468, 431]
# 测试集文件名列表
TEST_FILENAME = [
    "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
    "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
    "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
    "x31", "x32", "x33", "x34", "x35"
]

# 测试集每个文件长度
TEST_LABEL_AMOUNT = [523, 469, 465, 482, 505,
                     450, 509, 517, 508, 510,
                     457, 527, 506, 490, 498,
                     515, 400, 459, 487, 513,
                     510, 482, 527, 429, 510,
                     520, 498, 495, 470, 511,
                     557, 538, 473, 475, 483]
# ECG信号采样率  1秒 100个点
ECG_RAW_FREQUENCY = 100

ANN_DICT = {'A': 1, 'N': 0}


def ann2num(ann, ann_dict):
    num = []
    for a in ann:
        num.append(ann_dict[a])
    return num


# print(path)
ecg = None
sample = None
symbol = None
nums = None
print('start')
for i in range(len(TRAIN_FILENAME)):
    path = ROOT+TRAIN_FILENAME[i]
    tmpecg = wfdb.rdrecord(path).p_signal
    ann = wfdb.rdann(path, 'apn')
    tmpsymbol = ann.symbol  # 'A','N'
    tmpnum = ann2num(tmpsymbol, ANN_DICT)  # 'A','N'转1,0
    tmpsample = ann.sample  # 标注  对应  ecg信号的起点
    if (len(tmpecg)//6000)>len(tmpnum):
        tmpecg=tmpecg[:len(tmpnum)*6000]
        tmpnum=tmpnum[:len(tmpecg)//6000]
        print(i,'ecg',len(tmpecg)/6000,'num',len(tmpnum))
    else:
        tmpnum=tmpnum[:len(tmpecg)//6000]
        tmpecg=tmpecg[:len(tmpnum)*6000]
        print(i,'ecg',len(tmpecg)/6000,'num',len(tmpnum))
    tmpecg=torch.FloatTensor(tmpecg)
    tmpnum=torch.LongTensor(tmpnum)
    if i == 0:
        ecg = tmpecg
        nums = tmpnum
    else:
        ecg=torch.cat((ecg,tmpecg),dim=0)
        nums=torch.cat((nums,tmpnum),dim=0)
# print(ecg.size())
# print(ecg.size(0)/6000)
# print(nums.size())
ecg=ecg.view(17023,6000,1)
# ecg=ecg.permute(0,2,1) #[17023,1,6000]
ecg=ecg.squeeze() #为了匹配rnn模型 [17023,1,6000]=>[17023,6000]=>[17023,60,100]
ecg=ecg.view(17023,60,100)#[17023,60,100]

#分批加载训练
# #生成数据集
full_data=Data.TensorDataset(ecg,nums)
print(full_data)
# #数据集划分
train_data,test_data=Data.random_split(full_data,[13600,3423])
# print(test_data[:][1].size()) #[3423]
#数据加载
train_loader=Data.DataLoader(dataset=train_data,batch_size=32,shuffle=True)
test_loader=Data.DataLoader(dataset=test_data,batch_size=100)
#模型加载
cnn=Rnn()
#优化设置
optimizer=torch.optim.SGD(cnn.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()
#模型保存地址
model_dir='E:/deeplearning/myecgpro/model_save/rnn.pth'
#训练
print('training...')
maxacc=0
maxnum=0
maxtotal=0
for epoch in range(50):
    for i,(x,y) in enumerate(train_loader):
        x,y=Variable(x),Variable(y)
        # print('times:',i+1,'x size:',x.data.size(),'y size:',y.data.size())
        # 425个批次  每个批次32batch
        out=cnn(x)
        loss=loss_func(out,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch,i)
    #每10轮 测试准确度 保存最高的准确度模型
    if epoch%1==0:
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        epochacc=0
        epochnum=0
        epochtotal=0
        for i,(test_x,test_y) in enumerate(test_loader):
            test_out=cnn(Variable(test_x))
            pred_y=torch.max(test_out,1)[1].data.numpy().squeeze()
            test_y=test_y.data.numpy()
            # acc=sum(pred_y==test_y)/len(pred_y)
            epoch_acc_num=sum(pred_y==test_y)
            epoch_total_num=len(pred_y)
            epochnum=epochnum+epoch_acc_num
            epochtotal=epochtotal+epoch_total_num
            # print(acc)
            # epochacc=epochacc+acc

            print('current num:',epochnum,'max num:',maxnum)
        #模型保存
        if epochnum>maxnum :
            maxnum=epochnum
            maxtotal=epochtotal
            state={'net':cnn.state_dict(),'optim':optimizer.state_dict(),'epoch':epoch,'maxnum':maxnum,'maxtotal':maxtotal}
            torch.save(state,model_dir)
            print('success save!')
        elif epochnum<maxnum and epochnum/maxnum<=0.9 : 
            print('break circle!')
            break
#测试准确度
print('testing acc...',maxnum/maxtotal)
