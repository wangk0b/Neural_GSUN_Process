loc_gen = function(size){
  grid = as.integer(sqrt(size))
  locations = matrix(0,nrow = size, ncol = 2)
  count = 0
  for(i in 1:grid){
   for(j in 1:grid){
     x = (i - 0.5 + runif(1,-0.4, 0.4))/grid
     y = (j - 0.5 + runif(1,-0.4, 0.4))/grid 
     count = count + 1
     locations[count,] = c(x,y)
   
   }
  
  }
  return(locations)

}


matern = function(sigma,beta,nu,h){
 value = sigma/2^(nu - 1)/gamma(nu)*(h/beta)^nu*besselK(h/beta,nu)
 return(value)

}

















