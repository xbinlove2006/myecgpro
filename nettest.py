import torch
import torch.nn as nn
from torch.autograd import Variable

#rnn
'''
一段ecg  6000个点  对应一个标签
可以把这一段ecg分成60份，对应人1分钟大概心跳60次
[1,6000]=>[1,60,100]
rnn理解：
rnn 就是一个普通的3层神经网络与一个2层的神经网络组合而成，怎么组合呢？
3层神经网络（100,16,2）经过隐藏层之后 得到了n个结点（如隐藏层16个结点），在继续向输出层传播之前
和一个2层的神经网络（16，16）的结果相加（2个长度为16的向量相加） 得到的结果代替原来的3层神经网络的隐藏层结果来向输出层传播
2层神经网络的输入是上一次的输出  初始的输入可以全为0  即记忆层
相加后得到的16是本层记忆层的输出，作为下一层记忆层的输入
传播后得到的2分类结果 是每层的预测结果，对于本ecg信号来说  只需要最后一次的预测结果即可
'''
# rnn=nn.RNN(100,16)
# x=torch.randn(60,32,100)
# out,h=rnn(x)
# print(out.size())
# print('out:',out[-1].size())
# # print(out[-1])
# print('h:',h.size())
# # print(h)
# out=nn.Linear(16,2)(out)
# print(out.size())

lstm=nn.LSTM(100,16)
x=torch.randn(60,32,100)
out,(h,c)=lstm(x)
print(out.size())
print(h.size())
print(c.size())