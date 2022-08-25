import sys 
sys.path.append("..") 

import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn.nn import ResTanhCell
from torchesn import utils
import time

device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)
X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
X_data = torch.from_numpy(X_data).to(device)
Y_data = torch.from_numpy(Y_data).to(device)

trX = X_data[:5000]
print(trX.shape)
trY = Y_data[:5000]
tsX = X_data[5000:]
tsY = Y_data[5000:]

washout = [0]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trY_flat = utils.prepare_target(trY.clone(), [trX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size,spectral_radius=0.5)
    model.to(device)

    model(trX, washout, None, trY_flat)
    model.fit()
    output, hidden = model(trX, washout)
    print("Training error:", loss_fcn(output, trY[washout[0]:]).item())

    # Test
    # here it should be noted that the hidden only about the lattest i.e. the 
    # last time step! so if we want to proceed to predict, next we should only 
    # input the last prediction, not all the test prediction 
    output, tmph = model(tsX, [0], hidden)
#    tmp = output[-1:,:,:]  
#    for i in range(700):
#        tmp, tmph = model(tmp[-1:,:,:], [0], tmph)
#        print(i,":",tmp.item())
    
    print("Test error:", loss_fcn(output, tsY).item())
    print("Ended in", time.time() - start, "seconds.")
