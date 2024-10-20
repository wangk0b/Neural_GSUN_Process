import numpy as np
import torch
import os
#from proc_gen import sun_proc
from trans_neural_bayes import neural_bayes
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



#T = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


criterion = nn.MSELoss()
device = torch.device("cuda:0")
Net = neural_bayes()
Net.to(device)
optimizer = optim.Adam(Net.parameters(), lr=0.001)
#np.random.seed(1)
for epoch in range(int(1e10)):
 #sigma = np.random.uniform(0.99,1.01,1)
 sigma = np.array([2.0])
 #beta_1 = np.random.uniform(0.11,0.12,1)
 beta_1 = np.array([0.03])
 #beta_2 = np.random.uniform(0.03,0.05,1)
 beta_2 = np.array([0.3])
 #nu = np.random.uniform(0.28,0.31,1)
 nu_1 = np.array([0.5])
 ##########################
 nu_2 = np.array([1])
 #skew = np.random.uniform(-0.1,0.1,1)
 skew_1 = np.array([-0.7])
 #batch = np.random.randint(1,10,1)
 skew_2 = np.array([0.5])
 ###########################
 batch = np.array([10])
 y = np.concatenate((sigma,beta_1,nu,beta_2,skew),axis = 0)
 y = y.reshape(-1)
 #print(y.shape)
 y = torch.from_numpy(y).float().to(device)
 #print(y.shape)
 filename = 'Rscript gen_sample.R '+ str(sigma.item())+ " "+ str(beta_1.item())+ " " +str(nu.item())+ " "+ str(beta_2.item())+ " "+str(skew.item())+" "+str(batch.item())
 #os.system(filename)
 #exit()
 x = np.load("data_gen.npy")
 #print(x.shape)
 x_pad = np.zeros((10,1,32,32))
 if len(x.shape) == 4:
   for i in range(x.shape[0]):
       x_pad[i,0,:,:] = np.pad(x[i,0,:,:], 11, pad_with, padder=0)
 else:
     x[0,:,:] = np.pad(x[0,:,:], 11, pad_with, padder=0)
 x_pad = torch.from_numpy(x_pad).float().to(device)
 #print(x_pad.shape)
 yhat = Net(x_pad)
 #print(yhat.shape)
 optimizer.zero_grad()
 loss = criterion(yhat,y)
# f = open("log_loss.txt", "a")
# f.write(str(loss.item())+"\n")
# f.close()
 print(loss.item())
 print("Real")
 print(yhat)
 print("Predict")
 print(y)
 loss.backward()
 optimizer.step()

model_scripted = torch.jit.script(Net) # Export to TorchScript
model_scripted.save('model_scripted.pt')




