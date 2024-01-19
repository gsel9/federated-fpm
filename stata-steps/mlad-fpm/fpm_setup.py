import jax.numpy as jnp
#import stata_setup
# NOTE: to enable loading sfi when not running via stata
#stata_setup.config('/Applications/Stata/', 'mp', splash=False)
# provides access to stata macros
from sfi import Macro


def mlad_setup(M):
	"""Setup file. Useful as we need the derivatives 
	of the log(time) spline variables. These are needed within
	the log-likelihood function.

	The setup file is called once before the iterations start.
	"""
	dnsvars = Macro.getGlobal("dnsvars").split()
	dns = []
	for v in range(len(dnsvars)):
		dns.append(M[dnsvars[v]])

	dns.append(jnp.zeros((len(M[dnsvars[1]]), 1)))
	dns = (jnp.array(dns).squeeze(axis=2)).T
	M["dns"] = dns

	return M