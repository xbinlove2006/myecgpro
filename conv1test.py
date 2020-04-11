import torch
import torch.nn as nn
from Net import CNN5
from torch.autograd import Variable

i=torch.randn(32,6000,1) #[batch,6000,1]
print('输入:',i.size())
i=i.permute(0,2,1)      #[batch,1,6000]
print('转换后',i.size())
x=Variable(i)
#conv1d(输入通道，输出通道，核大小，步长，扩充)
conv1d=nn.Conv1d(1,100,5,1,2) #[batch,100,6000]  padding=(kernelsize-1)/2   2=(5-1)/2
pooling=nn.MaxPool1d(10)    #[batch,100,600]
y=conv1d(x)
print('卷积后',y.size())
y=pooling(y)
print('池化后',y.size())
print('-----------------------')
conv2=nn.Conv1d(100,10,5,1,2) #[batch,10,600]
pool=nn.MaxPool1d(10) #[batch,10,60]
y=conv2(y)
print('二次卷积后：',y.size())
y=pool(y)
print('二次池化后：',y.size())