from scipy.special import kv
from scipy.spatial import distance_matrix
from scipy.special import gamma
import numpy as np
from minimax_tilting_sampler import TruncatedMVN

class cov_generator:
 def __init__(self,size,spacing = 'ir'):
     self.size = size
     self.spacing = spacing 
     self.grid = int(np.sqrt(self.size))

 def LocGenXY(self):
    locations = []
    grid  = self.grid
    if self.spacing == "ir":
      for i in range(grid):
        for j in range(grid):
            x = (i + 0.5 + np.random.uniform(-0.4, 0.4))/grid
            y = (j + 0.5 + np.random.uniform(-0.4, 0.4))/grid
            locations.append([x,y])
    else:
         step = 1/grid
         x = np.arange(0,1,step) + step/10
         y = np.arange(0,1,step) + step/10
         locations = [[i,j] for i in x for j in y ]
    return np.array(locations)


 def dist_mat(self,locations):
    d = distance_matrix(locations,locations)
    return d



 def matern_cov(self,sigma,beta,nu,d_mat):
    n = d_mat.shape[0]
    value = np.ones((n,n))*sigma
    con = sigma/(2**(nu - 1)*gamma(nu))
    for i in range(n):
        for j in range(i+1,n):
         expr = d_mat[i,j]/beta
         value[i,j] = con*(expr**nu)*kv(nu,expr)
         value[j,i] = value[i,j]
    return value



def sun_proc(N,sigma,beta_1,beta_2,nu,skew,n_samp):
  s = cov_generator(N)
  loc = s.LocGenXY()
  d = s.dist_mat(loc)
  cov_one  = s.matern_cov(sigma,beta_1,nu,d)
  cov_two = s.matern_cov(1,beta_2,0.5,d)
  L_one = np.linalg.cholesky(cov_one)
  eigenvalues_1,eigenvectors_1 = np.linalg.eigh(cov_one)
  eigenvalues_2,eigenvectors_2 = np.linalg.eigh(cov_two)
  H = 0
  for i in range(N):
     H += eigenvalues_1[i]*eigenvectors_1[:,i] + eigenvalues_2[i]*eigenvectors_2[:,i]
  H = skew*H
  output = np.zeros((n_samp,3,s.grid,s.grid))
  loc_1 = loc[:,0].reshape((1,s.grid,s.grid))
  loc_2 = loc[:,1].reshape((1,s.grid,s.grid))
  mu = np.zeros(N)
  lb = np.zeros(N)
  ub = np.ones(N) * np.inf
  tmvn = TruncatedMVN(mu, cov_two, lb, ub)
  for j in range(n_samp):
   W = L_one @ np.random.normal(0,1,N)
   U = tmvn.sample(1)
   print(U.shape)
   exit()
  # print("+++++++++")
   X = W + H * U.reshape(-1)
   X = X.reshape((1,s.grid,s.grid))
   result = np.concatenate((X,loc_1,loc_2),axis = 0)
   output[j,:,:,:] = result
  return result
















