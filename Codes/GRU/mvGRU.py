# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import scipy
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt
device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)


#################### model defined ############################################
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout = 0):
        super(GRUModel,self).__init__()
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, dropout=dropout,\
                            batch_first=True)
      
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x,hidden=None):
        y, hn = self.gru(x,hidden)
        y = self.fc(y)
        return y[:,-1,:], hn

class sequenceDataset(torch.utils.data.Dataset):
    def __init__(self,X,Y,seq_len=50):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,i):
        if (i >= self.seq_len-1):
            i_start = i-self.seq_len+1
            x = self.X[i_start:i+1,:]
        else:
            padding = self.X[0,:].repeat(self.seq_len-i-1,1)
            x = self.X[0:(i+1),:]
            x = torch.cat((padding,x),0)
        return x, self.Y[i,:]
            

# loss test function
def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            output, _ = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test avg loss: {avg_loss}")
    return avg_loss

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        output, _ = model(X)
        
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    

###################### dataset prepared #######################################
input_size = 35
output_size = 35

#data = torch.sin(torch.Tensor(np.linspace(0,20,5000)))+torch.randn(5000)/100
#data = data.reshape(5000,1,1)
#print("whole data size:",data.shape)
#data = data.to(device)
#trdata = data[:4000,:,:]
#testdata = data[4000:,:]
#
#trX = trdata[:-1,:,:]
#trY = trdata[1:,:,:]
#
#tsX = testdata[:-1,:,:]
#tsY = testdata[1:,:,:]

Data = scipy.io.loadmat("G:/cfd_programe/project/transition_prediction/Datas/long/raw/400/subA.mat")["subA"]
Data = torch.Tensor(Data[:,:]).to(device)

critical = 11000
window = 100
if (critical>=Data.shape[0] or critical%window != 0):
    sys.exit(0)

X_data = Data[:-1, :input_size]
Y_data = Data[1:, :input_size]

trX = X_data[:critical,:]
trY = Y_data[:critical,:]
tsX = X_data[critical:,:]
tsY = Y_data[critical:,:]


train_dataset = sequenceDataset(trX,trY,window)
train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

test_dataset = sequenceDataset(tsX,tsY,window)
test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=False)

###################### training prepared #######################################
mv_net = GRUModel(input_size,70,2,output_size,0.00)
mv_net.double()
mv_net = mv_net.to(device)

print(mv_net)

criterion = nn.MSELoss() 
optimizer = optim.Adam(mv_net.parameters(),lr=0.0001)





###################### training #######################################

print("untrain network:")
test_model(test_loader,mv_net,criterion)

print("--------------- begin training ------------------------")
trainloss_list = []
testloss_list = []
xaxis = []
total_timestart = time.time()
total_epoch = 60
for epoch in range(total_epoch):
    total_loss = 0
    num_batches = len(train_loader)
    
    print(epoch)
    mv_net.train()
    start_time = time.time()
    if (epoch>30):
        optimizer.param_groups[0]["lr"] = 0.0001
    elif (epoch > 60):
        optimizer.param_groups[0]["lr"] = 0.00001
    for i, data in enumerate(train_loader,1):
        optimizer.zero_grad()
        xdata,ydata = data
        out, _ = mv_net(xdata)
        loss = criterion(out,ydata)
        total_loss += loss.item()
        loss.backward() #get the gradient of each variant
        optimizer.step()
        optimizer.zero_grad() #reset the gradient
        if (i%100==0):
            print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.item()))
    xaxis.append(epoch)
    if (epoch%4==0):
        torch.save(mv_net,"GRUlong{}.pth".format(epoch))
    print("\n epoch:{} time used:".format(epoch),-start_time+time.time())
    print("train avg loss:", total_loss/num_batches,",","train total loss: ", total_loss,)
    trainloss_list.append(total_loss/num_batches)
    testloss_list.append(test_model(test_loader,mv_net,criterion))

print("total time used:",time.time()-total_timestart)

###################### diagnose curve #######################################
plt.title("total epoch:{}, input:{}, output:{}, learning rate:{}".format(total_epoch,input_size,output_size,0.0001))
plt.plot(xaxis,trainloss_list,label="train loss")
plt.plot(xaxis,testloss_list,label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()

fig = plt.gcf()
fig.savefig("diagnose.png")
fig.clear()


###################### prediction #######################################