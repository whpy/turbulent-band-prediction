# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 14:55:15 2022

@author: WHPY5
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

class LSTMCellModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMCellModel,self).__init__()
        self.hidden_dim = hidden_dim
        
        self.layer_dim = layer_dim

        self.lstm = nn.LSTMCell(input_dim, hidden_dim,\
                            batch_first=True)
      
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self,x,hidden=None):
        
        

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