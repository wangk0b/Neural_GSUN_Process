seed_option =commandArgs(trailingOnly = TRUE)
options(digits=5)
library(reticulate)
library("tmvtnorm")
source("sample.R")
#print(seed_option)
#set.seed(1)
sigma = as.double(seed_option[1]) 
beta_1 = as.double(seed_option[2]) 
nu = as.double(seed_option[3]) 
beta_2 = as.double(seed_option[4]) 
skew = as.double(seed_option[5]) 
batch = strtoi(seed_option[6])

#loc = loc_gen(100)
x = seq(0,1,length = 10)
y = seq(0,1,length = 10)
loc = expand.grid(x,y)
d_mat = dist(loc,diag=T,upper=T)
out = array(0,dim = c(batch,1,10,10))
#p_out = array(0, dim = c(10000,5))

#for(j in 1:100){
#sigma_new = sigma + runif(1,-0.1,0.1)
#beta_1_new = beta_1 + runif(1,-0.005,0.005)
#nu_new = nu + runif(1,-0.05,0.05)
#beta_2_new = beta_2 + runif(1,-0.005,0.005)
#skew_new = skew + runif(1,-0.1,0.1)

#params = c(sigma_new,beta_1_new,nu_new,beta_2_new,skew_new)
#p_out[j,] = params

H_one = as.matrix(matern(sigma,beta_1,nu,d_mat))
diag(H_one) = sigma

H_two = as.matrix(matern(1,beta_2,0.5,d_mat))
diag(H_two) = 1

L_one = chol(H_one)

#print(sigma_new)

for(k in 1:batch){
W = L_one %*% rnorm(100)

H = 0
e_1 = eigen(H_one)
e_2 = eigen(H_two)

for(i in 1:100){
 H = H + e_1$values[i]*e_1$vectors[,i] + e_2$values[i]*e_2$vectors[,i]
 
}

H = skew * H
U = rtmvnorm(1, mean = rep(0, 100),sigma = H_two,lower=rep(0, 100),algorithm = "gibbs")
#print(c(U))
X = W + c(H) * c(U)
loc_1 = matrix(loc[,1],10,10)
loc_2 = matrix(loc[,2],10,10)
X = matrix(X,10,10)
#print(X)
#out = array(0,dim = c(3,20,20))
out[k,1,,] = X
#out[batch,2,,] = loc_1
#out[batch,3,,] = loc_2
}

np = import("numpy")
np$save("data_gen.npy",out)
#np$save("params.npy",p_out)




