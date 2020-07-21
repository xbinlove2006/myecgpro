#将CNN和残差网络的参数设置一致，来比较性能
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#batchnormalize cnn
class CNNNET_cnn5(nn.Module):
    def __init__(self):
        super(CNNNET_cnn5,self).__init__()
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
            nn.Linear(10*60,2)
            # nn.Linear(10*60,128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128,16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Linear(16,2)
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
            #[batch,ch_in,600]
            nn.Conv1d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1),#[batch,ch_out,600/stride]
            nn.BatchNorm1d(ch_out),
            nn.ReLU(),
            nn.Conv1d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)#[batch,ch_out,上一个600/stride]
        )
        self.shortcut=nn.Sequential()
        if ch_in!=ch_out:
            self.shortcut=nn.Sequential(
                nn.Conv1d(ch_in,ch_out,kernel_size=1,stride=stride),#[batch,ch_in,600]=>[batch,ch_out,600/stride]
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
            nn.Conv1d(1,100,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm1d(100),
            nn.MaxPool1d(kernel_size=10)#[batch,100,600]
            # nn.ReLU()
            #[batch,100,6000]
        )
        self.resblock=nn.Sequential(
            ResBlock(100,200,stride=2),#[batch,100,600]=>[batch,200,300]
            ResBlock(200,400,stride=2),#[batch,200,300]=>[batch,400,150]
            ResBlock(400,800,stride=2),#[batch,400,150]=>[batch,800,75]
            ResBlock(800,80,stride=1),#[batch,800,75]=>[batch,80,75]
            ResBlock(80,8,stride=1)#[batch,80,75]=>[batch,8,75] 8*75=600
        )
        self.fc=nn.Sequential(
            nn.Linear(8*75,2)
            # nn.Linear(512*375,128),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128,16),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Linear(16,2)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.resblock(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x