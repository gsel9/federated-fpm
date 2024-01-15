"""
// mlad is an alternative optimizer in Stata
// It calls Python and most calculations are performed within Python
// mlad requires a Python file to define the log-likelhood (fpm_hazard_ll.py)

// Note that gradient (score) and Hessian functions are automatically obtained
// when using mlad using automatic differentiation.

// There is an option to include a Python setup function
// It is useful here as we need the derivatives of the log(time) spline variables 
// These are needed within the log-likelihood function
// The setup function is called once before the iterations start 

// mlad is an alternative optimizer in Stata
// It calls Python and most calculations are performed within Python
// mlad requires a Python file to define the log-likelhood

// The Python file is only a few lines as the log-likelhood is simple type fpm_hazard_ll.py

// Note that gradient (score) and Hessian functions are automatically obtained
// when using mlad using automatic differentiation.

// There is an option to include a Python setup file
// It is useful here as we need the derivatives of the log(time) spline variables 
// These are needed within the log-likelihood function
// The setup file is called once before the iterations start. type fpm_setup.py
"""

import jax.numpy as jnp   
import mladutil as mu
from sfi import Macro


def python_ll(beta, X, wt, M):
	"""Use AD from Jax for.
 
 	Python then calculated the following.
        The value of the log likelihood (a scalar)
        The score (gradient) vector (a 1 x p vector)
        The Hessian (second derivatives) matrix (a p x p matrix).

	Args:
		beta (_type_): parameters
		X (_type_): data
		wt (_type_): weights (vector of 1's if no weights in mlad)
		M (_type_): dictionary containing othervars etc
	"""
	xb = mu.linpred(beta, X, 1)
	time = mu.linpred(beta, X, 2)
	eta = xb + time
	dtime = jnp.dot(M["dns"], beta[1])[:, None] 
	return jnp.sum(wt * (M["_d"] * (jnp.log(dtime) + eta) - jnp.exp(eta)))


def mlad_setup(M):
	dnsvars = Macro.getGlobal("dnsvars").split()
	dns = []
	for v in range(len(dnsvars)):
		dns.append(M[dnsvars[v]])

	dns.append(jnp.zeros((len(M[dnsvars[1]]),1)))
	dns = (jnp.array(dns).squeeze(axis=2)).T
	M["dns"] = dns
	return M