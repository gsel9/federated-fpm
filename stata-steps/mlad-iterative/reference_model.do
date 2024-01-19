insheet using "data.csv", comma clear

// stset declares the data in memory to be st data, informing 
// Stata of key variables and their roles in a survival-time 
// analysis
stset t, failure(d=1)

mlad (ln_lambda: = x1 x2)   ///
     (ln_gamma: ),          /// 
      othervars(_t _d)      ///
      othervarnames(t d)    ///
      llfile(weibull_ll) 
