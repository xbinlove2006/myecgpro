import wfdb
from Ecg import *
from Net import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
ROOT = 'apnea-ecg-database-1.0.0/'
ANN_DICT={'A':1,'N':0}
name = 'a01'

path = ROOT+name

rawecg = wfdb.rdrecord(path).p_signal
ann = wfdb.rdann(path,'apn').symbol
num=ann2num(ann,ANN_DICT)
# print(ann)
# print(num)
divided_sigs = div_sig(rawecg, len(ann))
# print(len(divided_sigs))  # 489
divided_sigs = torch.Tensor(divided_sigs)  # size[489,6000,1]
divided_sigs = divided_sigs.squeeze()  # size[489,6000]
num=torch.Tensor(num)
x=divided_sigs[50:]
y=num[50:]
x=x.type(torch.FloatTensor)
y=y.type(torch.LongTensor)
x,y=Variable(x),Variable(y)
net=MLP(6000,64,2)
optimizer=torch.optim.SGD(net.parameters(),lr=0.02)
loss_func=torch.nn.CrossEntropyLoss()

out=net(x)
loss=loss_func(out,y)
optimizer.zero_grad()#清空梯度
loss.backward()#计算梯度
optimizer.step()#更新权重参数


#dim=0:按列（每列最大值）
#dim=1:按行（每行最大值）
#输出是一个[0.5,0.9]的矩阵，所以要按行
#max返回：[值，索引],用max()[1]取最大值索引，得到预测的是0还是1
prediction=torch.max(out,dim=1)[1]
pred_y=prediction.data.numpy().squeeze()#去掉维度为1的维度=》将维度(1,3)->(3)
target_y=y.data.numpy()

accuracy=sum(pred_y==target_y)/439
print(accuracy)

test_x=divided_sigs[:50].type(torch.FloatTensor)
test_y=num[:50].type(torch.LongTensor)
test_x,test_y=Variable(test_x),Variable(test_y)
test_out=net(test_x)
pre=torch.max(test_out,dim=1)[1]
pre_y=pre.data.numpy().squeeze()
print('真实值：')
print(test_y.data.numpy())
print('预测值：')
print(pre_y)
print(sum(pre_y==test_y.data.numpy())/50)