import numpy as np
import torch
import os
#from proc_gen import sun_proc
from trans_neural_bayes import LeNet5
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms



#T = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])

#def pad_with(vector, pad_width, iaxis, kwargs):
#    pad_value = kwargs.get('padder', 10)
#    vector[:pad_width[0]] = pad_value
#    vector[-pad_width[1]:] = pad_value
y_test = np.array([1,0.15,1,0.1,0.5,0.55,-0.3])
criterion = nn.MSELoss()
device = torch.device("cuda:0")
y_test = torch.from_numpy(y_test).float().to(device)

#load the trained networks
#Net = torch.jit.load('CNN_scripted.pt')
Net = torch.jit.load('GAT_encoder_scripted.pt')
Net.to(device)
optimizer = optim.SGD(Net.parameters(), lr=0.1)
#np.random.seed(1)
for epoch in range(int(10)):
 sigma = np.random.uniform(0.3,3,1)
 #sigma = np.array([0.8])
 beta_1 = np.random.uniform(0.01,1,1)
 #beta_1 = np.array([0.1])
 beta_2 = np.random.uniform(0.01,1,1)
 #beta_2 = np.array([0.1])
 nu_1 = np.random.uniform(0.3,2,1)
 #
 nu_2 = np.random.uniform(0.3,2,1)
 #nu = np.array([0.5])
 skew_1 = np.random.uniform(-3,3,1)
 skew_2 = np.random.uniform(-3,3,1)
 #skew = np.array([0.55])
 #print("One success!")
 #batch = np.random.randint(1,10,1)
 batch = np.array([10])
 y = np.concatenate((sigma,beta_1,nu_1,beta_2,nu_2,skew_1,skew_2),axis = 0)
 y = y.reshape(-1)
 #print(y.shape)
 y = torch.from_numpy(y).float().to(device)
 #print(y.shape)
 filename = 'Rscript gen_sample.R '+ str(sigma.item())+ " "+ str(beta_1.item())+ " " +str(nu_1.item())+ " "+ str(beta_2.item())+ " "+ str(nu_2.item())+ " "+str(skew_1.item())+" "+str(skew_2.item())+ " "+str(batch.item())
 os.system(filename)
 #exit()
 x = np.load("data_gen_test_new.npy").reshape((100,10,3,10,10))
# print(x.shape)
# x_pad = np.zeros((10,1,32,32))
# if len(x.shape) == 4:
#   for i in range(x.shape[0]):
#       x_pad[i,0,:,:] = np.pad(x[i,0,:,:], 11, pad_with, padder=0)
# else:
#     x[0,:,:] = np.pad(x[0,:,:], 11, pad_with, padder=0)
 x = torch.from_numpy(x).float().to(device)
 #print(x_pad.shape)
 for t in range(10):
  yhat = Net(x[t,:,:,:,:])
 #print(yhat.shape)
  optimizer.zero_grad()
  loss = criterion(yhat,y_test)
# f = open("log_loss.txt", "a")
# f.write(str(loss.item())+"\n")
# f.close()
 print(loss.item())
 print("Real")
 print(y)
 print("Predict")
 print(yhat)
 loss.backward()
 optimizer.step()
 test_x = np.load("data_gen_test_new.npy")
 #print(test_x.shape)
 test_x = np.load("data_gen_test_new.npy").reshape((100,10,3,10,10))
 #l = test_x.shape[0]
 Net.eval()
 loss_test = 0
 y_hold = 0
 for t in range(100):
     x_test = torch.from_numpy(test_x[t,:,:,:,:]).float().to(device)
     y_test_hat = Net(x_test) 
     y_hold += y_test_hat
     loss_test += criterion(y_test_hat,y_test) 
 print(y_hold/100)
 print(loss_test.item()/100)
 #if loss_test.item()/100 < 0.005:
 #   Net.train()
 #   optimizer = optim.SGD(Net.parameters(), lr=0.001)
 #if loss_test.item()/100 < 0.002:
 #    model_scripted = torch.jit.script(Net) 
 #    model_scripted.save('model_scripted.pt')
 #    exit()
# Net.train()

model_scripted = torch.jit.script(Net) # Export to TorchScript
model_scripted.save('model_scripted.pt')




