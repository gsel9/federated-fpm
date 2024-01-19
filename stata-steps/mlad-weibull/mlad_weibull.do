// Simulate some survival data 

set obs 500000
set seed 987123

gen x1 = rnormal()
gen x2 = rnormal()

// Simulation of survival data
survsim t d, dist(weibull) lambda(0.2) gamma(0.8) maxt(5) cov(x1 0.1 x2 0.1)

// stset declares the data in memory to be st data, informing 
// Stata of key variables and their roles in a survival-time 
// analysis
stset t, failure(d=1)

mlad (ln_lambda: = x1 x2)   ///
     (ln_gamma: ),          /// 
      othervars(_t _d)      ///
      othervarnames(t d)    ///
      llfile(weibull_ll) 

// fixing the number of steps in the optimizer
set maxiter 1

mlad (ln_lambda: = x1 x2)   ///
     (ln_gamma: ),          /// 
      othervars(_t _d)      ///
      othervarnames(t d)    ///
      llfile(weibull_ll) 
