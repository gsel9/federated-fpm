// Load data
use https://www.pclambert.net/data/rott3, clear

// Use stpm3 model to fit simple model in Stata
// Variables
//   -- hormon (factor variable) 
//   -- age (natural spline)
stpm3  i.hormon @ns(age,df(3)), scale(lncumhazard) df(4) 

estimates store stpm3

// Here is a simple way to get initial values
// fit exponential model & use least squares
// I will use the spline variables created by stpm3 when I use mlad

streg i.hormon _ns_f1_age1 _ns_f1_age2 _ns_f1_age3, dist(exp)

predict surv, surv

gen logcumH = log(-log(surv))

// Store inital values
matrix b_init = e(b)

// mlad is an alternative optimizer in Stata
// It calls Python and most calculations are performed within Python
// mlad requires a Python file to define the log-likelhood
// Fit model using mlad
// need to supply the two python files
// Setup two equations as stpm3
//   -- xb equation is for covariates effects
//   -- time equation is for effect of time
// Also the event indicator (_d) and derivatives (_dns1 _dns2 _dns3 _dns4) of
// the spline variables are needed

global dnsvars _dns1 _dns2 _dns3 _dns4

mlad (xb:   = i.hormon _ns_f1_age1 _ns_f1_age2 _ns_f1_age3, nocons ) ///
     (time: = _ns1 _ns2 _ns3 _ns4),                                  ///
      othervars(_d _dns1 _dns2 _dns3 _dns4)                          ///
      pysetup(fpm_setup)                                             ///
      llfile(fpm_hazard_ll)                                          ///
      init(b_init) search(off)                                                  

estimates store mlad       

// display estimates       
ml display

// compare estimates and standard errors
estimates table stpm3 mlad, se
 
// Note: leads to identical estimates
// mlad is faster in large datasets (depending on number of cores)
log close    
