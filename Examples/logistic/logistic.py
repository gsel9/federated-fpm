import jax.numpy as jnp   
import mladutil as mu

# Main file - sends parameters to each center
def python_ll(beta,X,wt,M):
  # should just send betas, but here data for each center is stored in M
  ll1 = GetCenter1(beta,M)
  ll2 = GetCenter2(beta,M)
  ll3 = GetCenter3(beta,M)
  return(ll1+ll2+ll3)

# each center would have an identical function to return log likelhood contribution
# each using the data stored at that center and the shared betas.  
  
# center 1 uses current betas to return sum of log likelihood  
def GetCenter1(beta,M):  
  X = M["X_center1"]
  y = M["y_center1"]
  xb  = mu.linpred(beta,X,1)
  return(jnp.sum(y*xb - jnp.log(1+jnp.exp(xb))))

# center 2 uses current betas to return sum of log likelihood  
def GetCenter2(beta,M):  
  X = M["X_center2"]
  y = M["y_center2"]
  xb  = mu.linpred(beta,X,1)
  return(jnp.sum(y*xb - jnp.log(1+jnp.exp(xb))))

# center 3 uses current betas to return sum of log likelihood  
def GetCenter3(beta,M):  
  X = M["X_center3"]
  y = M["y_center3"]
  xb  = mu.linpred(beta,X,1)
  return(jnp.sum(y*xb - jnp.log(1+jnp.exp(xb))))
