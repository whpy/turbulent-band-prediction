import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
import os
import scipy.io
from sklearn import preprocessing


device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = scipy.io.loadmat("G:/ML/ESN/data/678-2_200/subA.mat")["subA"]
elif dtype == torch.float:
    data = scipy.io.loadmat("G:/ML/ESN/data/678-2_200/subA.mat")["subA"]
    
# input (seq_len, batch, input_size)
input_size = 10
feature = 2
output_size = input_size+feature
data = data[:,:input_size]

Re = np.ones((data.shape[0],1))
data = np.c_[data,Re]
eny = np.zeros((data.shape[0],1))
for i in range(eny.shape[0]):
    eny[i,0] = np.linalg.norm(data[i,:])/1.76
data = np.c_[data,eny.squeeze()]




print(data.shape)
#X_data = np.expand_dims(data[:-1, :input_size], axis=1)
#Y_data = np.expand_dims(data[1:, :output_size], axis=1)
X_data = np.expand_dims(data[:-1, :input_size], axis=1)
Y_data = np.expand_dims(data[1:, :output_size], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:4000]
trY = Y_data[:4000]
tsX = X_data[4000:]
tsY = Y_data[4000:]
print("data prepared")

washout = [0]

hidden_size = 600
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size,nonlinearity='tanh',\
                leaking_rate=0.95,spectral_radius=0.95,num_layers=1,lambda_reg=0.05,\
                density=0.2,w_io=False)
    #model = ESN(input_size, hidden_size,output_size,nonlinearity='tanh',leaking_rate=0.95,lambda_reg=0.05,num_layers=1, spectral_radius=0.5,density=0.2)
    model.to(device)
    print("model prepared")
    
    model(trX, washout, None, trY_flat)
    model.fit()
    tr_output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(tr_output, trY[washout[0]:]).item())

    # Test
    # here it should be noted that the hidden only about the lattest i.e. the 
    # last time step! so if we want to proceed to predict, next we should only 
    # input the last prediction, not all the test prediction 
    ts_output, tmph = model(tsX, [0], hidden)
    
    # Prediction
    # we select the part that saddle point may exist1
    tmp_output,tmph = model(X_data[3250:4000],washout)
    tmp = tmp_output[-1:,:,:]  
    print(tmp.shape)
    
    preddim = 4000
    prediction = torch.ones(preddim,input_size).to(device)
    for i in range(preddim):
        tmp, tmph = model(tmp[-1:,:,:input_size], [0], tmph)
        prediction[i,:] = tmp[-1:,:,:input_size]
        if (i%1000==0):
            print(i)
    prefix = input("type a prefix:")
    if (prefix != "null"):
        np.savetxt("{}pred{}.csv".format(prefix,preddim),prediction.detach().cpu().numpy(),delimiter=",")
    print("Training error:", loss_fcn(tr_output, trY[washout[0]:]).item())
    print("Test error:", loss_fcn(ts_output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")
