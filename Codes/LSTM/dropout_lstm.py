# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 21:42:42 2022

@author: WHPY5
"""
import os
import scipy
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

if dtype == torch.double:
    data = scipy.io.loadmat("G:/ML/ESN/data/678-2_200/subA.mat")["subA"]
elif dtype == torch.float:
    data = scipy.io.loadmat("G:/ML/ESN/data/678-2_200/subA.mat")["subA"]

input_size = 10
feature = 0
output_size = input_size+feature
Data = data[:,:input_size]

X_data = Data[:-1, :input_size]
Y_data = Data[1:, :input_size]
#X_data = torch.from_numpy(X_data).to(device)
#Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:4000]
trY = Y_data[:4000]
tsX = X_data[4000:]
tsY = Y_data[4000:]

train_x = np.zeros((3900,100,input_size))
train_y = np.zeros((3900,100,input_size))

tsY = np.expand_dims(tsY, axis=0)
tsX = np.expand_dims(tsX, axis=0)
tsY = torch.from_numpy(tsY).to(device)
tsX = torch.from_numpy(tsX).to(device)
print("test data OK")
for i in range(0,3900):
    train_x[i,:,:] = trX[i:i+100,:]
    train_y[i,:,:] = trY[i:i+100,:]
train_x = torch.from_numpy(train_x).to(device)
train_y = torch.from_numpy(train_y).to(device)
print("slide windows OK")
trX = torch.from_numpy(np.expand_dims(trX,axis=0)).to(device)
trY = torch.from_numpy(np.expand_dims(trY,axis=0)).to(device)
print("train data OK")

batch_size = 64
train_ids = TensorDataset(train_x,train_y)
train_loader = DataLoader(dataset=train_ids,batch_size=batch_size,shuffle=True)
print("dataloader prepared,batch size = {}".format(batch_size))


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,dropout = 0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, dropout=dropout,\
                            batch_first=True)
      
        self.fc = nn.Linear(hidden_dim, output_dim)


    def forward(self, x, hidden=None):
        x, (hn, cn) = self.lstm(x, hidden)
        x = self.fc(x) 
        return x, (hn,cn)

#mv_net = LSTMModel(input_size,70,2,output_size,0.00)
mv_net = torch.load("MBGD80.pth")
mv_net.lstm.dropout = 0.0
mv_net.double()
mv_net = mv_net.to(device)

print(mv_net)

criterion = nn.MSELoss() 
optimizer = optim.Adam(mv_net.parameters(),lr=0.001)
#optimizer = optim.SGD(mv_net.parameters(),lr=0.0001,momentum=0.9)

trainloss = []
testloss = []
totstart = time.time()
for epoch in range(81,121):
    print(epoch)
    start = time.time()
    mv_net.train()
    if (epoch>40):
        optimizer.param_groups[0]["lr"] = 0.0001
    elif (epoch > 60):
        optimizer.param_groups[0]["lr"] = 0.00001
    for i, data in enumerate(train_loader,1):
        optimizer.zero_grad()
        xdata,ydata = data
        out,(_,_) = mv_net(xdata)
        loss = criterion(out,ydata)
        loss.backward() #get the gradient of each variant
        optimizer.step()
        optimizer.zero_grad() #reset the gradient
        if (i%100==0):
            print("epoch:{}, iter:{}, loss:{}".format(epoch,i,loss.item()))
    
    mv_net.eval()
    with torch.no_grad():
        tr_out,(_,_) = mv_net(trX)
        trloss = criterion(tr_out,trY)
        trainloss.append(trloss.item())
        print("train loss:",trloss.item())
    with torch.no_grad():    
        ts_output,(_,_) = mv_net(tsX)
        tsloss = criterion(ts_output,tsY)
        testloss.append(tsloss.item())
        print("test loss",tsloss.item())
    if (epoch%20==0):
        torch.save(mv_net,"MBGD{}.pth".format(epoch))
    print("{} second time used".format(time.time()-start))
print("total time used:{}".format(time.time()-totstart))
prefix0 = input("test prefix")
torch.save(testloss,"{}testloss".format(prefix0))
prefix1 = input("train prefix")
torch.save(trainloss,"{}trainloss".format(prefix1))


################## prediction ##############################
predsteps = 4000
xtmp = Data[3250:4000,:input_size]
xtmp = torch.tensor(np.expand_dims(xtmp,0))
xtmp = xtmp.to(device)
pred = torch.zeros(predsteps,input_size)
pred = pred.to(device)

with torch.no_grad():
    xtmp,(htmp,ctmp) = mv_net(xtmp)
    tmp = xtmp[:,-1:,:]
    for i in range(predsteps):
        tmp,(htmp,ctmp) = mv_net(tmp,(htmp,ctmp))
        pred[i,:] = tmp[0,-1,:]
        if (i%100==0):
            print(i," step")

pred = pred.detach().cpu().numpy()
np.savetxt("pptlstmpred.csv",pred,delimiter=",")
np.savetxt("pptlstmtr.csv",tr_out.detach().cpu().numpy()[0,:,:],delimiter=",")