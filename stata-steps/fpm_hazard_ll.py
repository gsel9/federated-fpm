import jax.numpy as jnp   
# NOTE: to enable loading mladutil when not running via stata
# stata_setup.config('/Applications/Stata/', 'mp', splash=False)
import mladutil as mu


def python_ll(beta, X, wt, M):
     """MLAD likelihood function. 

     Used to obtain the gradient (score) and Hessian functions 
     using automatic differentiation.

     Args:
     . //   -- beta - model parameters (coefficients)
     . //   -- X    - data
     . //   -- wt   - weights (vector of 1's if no weights in mlad)
     . //   -- M    - dictionary containing othervars etc
     """
     xb   =  mu.linpred(beta, X, 1)
     time =  mu.linpred(beta, X, 2)
     eta = xb + time
     dtime = jnp.dot(M["dns"], beta[1])[:, None] 
     
     return jnp.sum(wt * (M["_d"] * (jnp.log(dtime) + eta) - jnp.exp(eta)))
