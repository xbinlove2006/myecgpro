import torch
from Net import MLP
from torch.autograd import Variable
import matplotlib.pyplot as plt
mlp=MLP(1,100,1)
print(mlp)
x=torch.linspace(-1,1,100) #[100]
print(x.size())
x=torch.unsqueeze(x,dim=1) #[100,1]
print(x.size())
# print(x)

y=x**2+0.2*torch.rand(x.size()) #[100,1]
print(y.size())
x,y=Variable(x),Variable(y)
optimizer=torch.optim.SGD(mlp.parameters(),lr=0.4,weight_decay=0.01)
loss_fun=torch.nn.MSELoss()

# y_pre1=mlp(x)
# print('y_pre1')
# print('size:',y_pre1.size())
# print('data:',y_pre1.data.numpy())
# loss=loss_fun(y_pre1,y)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# y_pre2=mlp(x)
# print('y_pre2')
# print('size:',y_pre2.size())
# print('data:',y_pre2.data.numpy())
loss_arr=[]
plt.ion()
for i in range(500):
    y_pre=mlp(x)
    loss=loss_fun(y_pre,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_arr.append(loss.item())
    if i%5==0:
        print('loss:',loss.item())
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),y_pre.data.numpy(),'r-',lw=5)
        plt.text(0.5,0,'loss=%f'%loss.item())
        plt.pause(0.2)
plt.ioff()
plt.show()
plt.plot(loss_arr)
plt.show()