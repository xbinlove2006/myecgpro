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