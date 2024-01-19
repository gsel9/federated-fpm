// Returns the log-likelhood for fitting a Weibull model with 
// the Stata ml optimizer.

// This is a d0 evaluator which means that only the 
// log-likelihood (a scalar) needs to be returned.

program weibull_d0
  version 17.0
  args todo b lnf g H
  
  tempvar lnlambda lngamma
  mleval `lnlambda' = `b', eq(1)
  mleval `lngamma'  = `b', eq(2)
  
  mlsum `lnf' = _d*(`lnlambda' + `lngamma' + (exp(`lngamma') - 1)*ln(_t)) - ///
                exp(`lnlambda')*_t^(exp(`lngamma')) 
  if (`todo'==0 | `lnf'>=.) exit
end