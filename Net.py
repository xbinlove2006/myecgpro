import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self,inputs,hiddens,outputs):
        super(MLP,self).__init__()
        self.model=torch.nn.Sequential(
            torch.nn.Linear(inputs,hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddens,outputs)
        )
    def forward(self,x):
        x=self.model(x)
        return x


class CNN5(nn.Module):
    def __init__(self):
        super(CNN5,self).__init__()
        self.conv1=nn.Sequential(
            # [batch,1,6000]
            nn.Conv1d(1,100,kernel_size=5,stride=1,padding=2),#[batch,100,6000]
            nn.MaxPool1d(kernel_size=10)#[batch,100,600]
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(100,10,kernel_size=5,stride=1,padding=2),#[batch,10,600]
            nn.MaxPool1d(10)#[batch,10,60]
        )
        self.fc=nn.Sequential(
            nn.Linear(10*60,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        batch=x.size(0)
        x=x.view(batch,10*60)
        return self.fc(x)

class CNN7(nn.Module):#epoch=3 0.65左右
    def __init__(self):
        super(CNN7,self).__init__()
        self.conv1=nn.Sequential(
            # [batch,1,6000]
            nn.Conv1d(1,128,kernel_size=3,stride=1,padding=1),#[batch,128,6000]
            nn.MaxPool1d(kernel_size=5)#[batch,128,1200]
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1),#[batch,256,1200]
            nn.MaxPool1d(2)#[batch,256,600]
        )
        self.conv3=nn.Sequential(
            nn.Conv1d(256,16,kernel_size=3,stride=1,padding=1),#[batch,16,600]
            nn.MaxPool1d(2)#[batch,16,300]
        )
        self.conv4=nn.Sequential(
            nn.Conv1d(16,10,kernel_size=3,stride=1,padding=1),#[batch,10,600]
            nn.MaxPool1d(10)#[batch,10,60]
        )
        self.fc=nn.Sequential(
            nn.Linear(10*60,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        batch=x.size(0)
        x=x.view(batch,10*60)
        return self.fc(x)



#batchnormalize cnn
class BN_CNN5(nn.Module):
    def __init__(self):
        super(BN_CNN5,self).__init__()
        self.conv1=nn.Sequential(
            # [batch,1,6000]
            nn.Conv1d(1,100,kernel_size=5,stride=1,padding=2),#[batch,100,6000]
            nn.BatchNorm1d(100),
            nn.MaxPool1d(kernel_size=10)#[batch,100,600]
        )
        self.conv2=nn.Sequential(
            nn.Conv1d(100,10,kernel_size=5,stride=1,padding=2),#[batch,10,600]
            nn.BatchNorm1d(10),
            nn.MaxPool1d(10)#[batch,10,60]
        )
        self.fc=nn.Sequential(
            nn.Linear(10*60,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        batch=x.size(0)
        x=x.view(batch,10*60)
        return self.fc(x)

#残差网络
#残差网络块
class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):#设置stride可以缩小维度大小，避免通道增加导致的参数急剧变大
        super(ResBlock,self).__init__()
        self.block=nn.Sequential(
            #[batch,ch_in,6000]
            nn.Conv1d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),#[batch,ch_out,6000/stride]
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)#[batch,ch_out,6000/stride]
        )
        self.shortcut=nn.Sequential()
        if ch_in!=ch_out:
            self.shortcut=nn.Sequential(
                nn.Conv1d(ch_in,ch_out,kernel_size=1,stride=stride),#[batch,ch_in,6000]=>[batch,ch_out,6000/stride]
                nn.BatchNorm1d(ch_out)
            )
    def forward(self,x):
        out=self.block(x)
        x=self.shortcut(x)
        return out+x

#18层残差网络
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        #[batch,1,6000]
        self.conv=nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
            #[batch,32,6000]
        )
        self.resblock=nn.Sequential(
            ResBlock(32,64,stride=2),#[batch,32,6000]=>[batch,64,3000]
            ResBlock(64,128,stride=2),#[batch,64,3000]=>[batch,128,1500]
            ResBlock(128,256,stride=2),#[batch,128,1500]=>[batch,256,750]
            ResBlock(256,512,stride=2)#[batch,256,750]=>[batch,512,375]
        )
        self.fc=nn.Sequential(
            nn.Linear(512*375,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16,2)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.resblock(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x


class Rnn(nn.Module):
# 一段ecg  6000个点  对应一个标签
# 可以把这一段ecg分成60份，对应人1分钟大概心跳60次
# [1,6000]=>[1,60,100]
# rnn理解：
# rnn 就是一个普通的3层神经网络与一个2层的神经网络组合而成，怎么组合呢？
# 3层神经网络（100,16,2）经过隐藏层之后 得到了n个结点（如隐藏层16个结点），在继续向输出层传播之前
# 和一个2层的神经网络（16，16）的结果相加（2个长度为16的向量相加） 得到的结果代替原来的3层神经网络的隐藏层结果来向输出层传播
# 2层神经网络的输入是上一次的输出  初始的输入可以全为0  即记忆层
# 相加后得到的16是本层记忆层的输出，作为下一层记忆层的输入
# 传播后得到的2分类结果 是每层的预测结果，对于本ecg信号来说  只需要最后一次的预测结果即可

# pytorch 中的RNN层会输出每一次的out，和最后一次的h(t)  所以在设计rnn类时要将RNN的out再经过一个全连接层即可
    def __init__(self):
        super(Rnn,self).__init__()
        self.rnn=nn.Sequential(
            nn.RNN(input_size=100,hidden_size=128,num_layers=1,batch_first=True)
        )
        self.fc=nn.Linear(128,2)
        self.fc2=nn.Linear(60*2,2)

    def forward(self,x):
        out,h=self.rnn(x)
        # out=self.fc(out[:,-1,:])# 因为默认out[32,60,2] 只需要用到最后一个out
        #仅适用最后一个out 效果极差  只有60-65%
        #尝试把所有out[batch,60,128]聚合
        # out=self.fc(out) #[batch,60,2]
        # out=out.view(out.size(0),-1)#[batch,120]
        # out=self.fc2(out)
        #效果64%

        
        return out


class Rnn_layer5(nn.Module):
    def __init__(self):
        super(Rnn_layer5,self).__init__()
        self.rnn=nn.Sequential(
            nn.RNN(input_size=100,hidden_size=512,num_layers=5,batch_first=True)
        )
        self.fc=nn.Linear(512,2)
    def forward(self,x):
        out,h=self.rnn(x)
        out=self.fc(out[:,-1,:])# 因为默认out[32,60,2] 只需要用到最后一个out
        return out

class Lstm_layer5(nn.Module):
    def __init__(self):
        super(Lstm_layer5,self).__init__()
        self.lstm=nn.Sequential(
            nn.LSTM(100,512,num_layers=5,batch_first=True)
        )
        self.fc=nn.Sequential(
                nn.Linear(512,128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128,32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16,2)
            )
    def forward(self,x):
        out,(h,c)=self.lstm(x)
        out=self.fc(out[:,-1,:])# 因为默认out[32,60,2] 只需要用到最后一个out
        return out
    pass


def main():
    ecg=torch.randn(32,60,100)
    rnn=Rnn()
    out=rnn(ecg)
    print(out.size())
    pass


if __name__ == "__main__":
    main()