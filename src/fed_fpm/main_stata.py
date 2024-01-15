"""
- run training iterations at the server 
- clients have local data and code 
	- local client code runs a stata program and a python program alternatingly 
		- 1. python program 
		- 2. stata program 
	- optimization steps 
		- initialize parameter estimates at the server 
		- at each node calculate in python
		    - the value of the log likelihood (a scalar)
		    - the score (gradient) vector (a 1 x p vector)
		    - the Hessian (second derivatives) matrix (a p x p matrix).
  		- at each node 
    		- pass variables calculated in python to stata
			- update the parameter estimates in stata
   				- one step of the Newton-Raphson algorithm 
  			- return the updated vector of parameters to the server 
		- at the server, aggregate the parameters from each client 

Dataset: https://www.pclambert.net/data/rott3

Use stpm3 model to fit simple model in Stata
Variables
-- hormon (factor variable) 
-- age (natural spline)
>>> stpm3  i.hormon @ns(age,df(3)), scale(lncumhazard) df(4) 

I will use the spline variables created by stpm3 when I use mlad

Here is a simple way to get initial values: fit exponential model & use least squares
>>> streg i.hormon _ns_f1_age1 _ns_f1_age2 _ns_f1_age3, dist(exp)

Exponential PH regression
>>> predict surv, surv
>>> gen logcumH = log(-log(surv))
>>> regress logcumH i.hormon _ns_f1_age1 _ns_f1_age2 _ns_f1_age3 _ns1 _ns2 _ns3 _ns4 if _d

Store inital values
>>> matrix b_init = e(b)


. // Fit model using mlad
. // need to supply the two python files
. // Setup two equations as stpm3
. //   -- xb equation is for covariates effects
. //   -- time equation is for effect of time
. // Also the event indicator (_d) and derivatives (_dns1 _dns2 _dns3 _dns4) of
. // the spline variables are needed
. global dnsvars _dns1 _dns2 _dns3 _dns4

. mlad (xb:   = i.hormon _ns_f1_age1 _ns_f1_age2 _ns_f1_age3, nocons ) ///
>      (time: = _ns1 _ns2 _ns3 _ns4)                                   ///
>      , othervars(_d _dns1 _dns2 _dns3 _dns4)                         ///
>        pysetup(fpm_setup)                                            ///
>        llfile(fpm_hazard_ll)                                         ///
>        init(b_init) search(off)      
"""


for i in range(n_client_steps):
	
	# per client do...
	# in python 
	ll_c1, grad_c1, hessian_c1 = likelihood_step(beta_i, data_c1, weights, metadata)
	# call to stata optimizer to update parameters 
 	beta_c1 = newton_raphs(beta_i, ll_c1, grad_c1, hessian_c1)
	### 
