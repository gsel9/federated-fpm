# Create some data (saved to disk in a csv)
# >>> stata -e data_generation.do 
# run mlad until convergence 
# >>> stata -e reference_model.do 
# run stepwise model
# 0. Generate initial parameter values 
# 1. For each server iteration and client do
#    - compute ML derivatives etc from with current parameters 
#    - pass derivatives etc to stata optimizer 
#    - output the current parameter estimates 
#    - repeat 
# >>> python stepwise_model.py

import stata_setup
# initialize stata env within python 
stata_setup.config('/Applications/Stata/', 'mp', splash=False)

from pystata import config, stata
# write stata output to a log file
config.set_output_file(filename='stepwise_model.log', replace=True)

import numpy as np
from jax import grad, jacfwd
from weibull_ll import python_ll

beta_i = np.random.random(3)

X = np.random.random((10, 3))
M = {
     "d": np.random.choice([0, 1], size=10),
     "t": np.random.random(10)
}

for server_iter in range(1):
     
     # 
     # --- likelihood computations in python
     # 
     ll = python_ll(beta=beta_i, X=X, wt=None, M=M)
     print(ll)
     dl_dbeta = grad(python_ll, argnums=0)(beta_i, X, None, M)
     print(dl_dbeta)
     
     d2l_dbeta2 = jacfwd(grad(python_ll, argnums=0)(beta_i, X, None, M)) 
     print(d2l_dbeta2)
     #d2l_dbeta2 = grad(dldx, argnums=0) #(beta_i, X, None, M)
     #print(d2l_dbeta2(beta_i, X, None, M))
     
     # 
     # --- optimization step in stata 
     # 
