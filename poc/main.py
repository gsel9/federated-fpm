""" Do in terminal 

>>> python -m pip install pystata stata-setup

The pystata Python package is shipped with Stata and located 
in the pystata subdirectory of the utilities folder in Stata’s 
installation directory. You can type the following in Stata to 
view the name of this directory:

   . display c(sysdir_stata)

Add to .zshrc (alternatively: ~/.bashrc or ~/.bash_profile)
    
    export PYTHONPATH="/Applications/Stata/utilities:$PYTHONPATH"
    
"""


# initialize stata env within python 
import stata_setup
stata_setup.config('/Applications/Stata/', 'mp', splash=False)

# to run stata cmds and create log file 
from pystata import stata, config 
config.set_output_file(filename='output.log', replace=True)

import numpy as np
import pandas as pd 

from jax import grad, jacfwd
from weibull_ll import logl


data = pd.read_csv("./data.csv")
print(data.head())

# data matrix 
X = data[["x1", "x2"]].values 

# event data 
M = {
     "d": data["d"].values,
     "t": data["t"].values
}

# initalize beta coefficients 
coefs = np.random.random(X.shape[1])
print(coefs)

# iterate over clients 
for _ in range(4):
    
    # compute likelihood 
	ll = logl(beta=coefs, X=X, wt=None, M=M)

	# derivative of likelihood 
	dl_dbeta = grad(logl, argnums=0)(coefs, X, None, M)
	print(dl_dbeta)
 
	# TODO:
	# - set init params to stata optimizer 
 	# - likelihood Hessian   
  	# - pass params (l, grads, ...) to stata opt
	# - do one step with stata opt
	# - get current coefs from stata opt  
 
	# run stata command
	cmd = """use "/Users/sela/Desktop/stata-data/example_data.dta", clear 
	stpm2 sex rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3  if site23 == 4, scale(hazard) bhazard(rate) iterate(50)  df(4) tvc(rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3) dftvc(2)"""

	stata.run(cmd)
    
# close the log file 
config.close_output_file()
