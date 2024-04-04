clear all
cd ${DRIVE}/GitHub/federated-fpm/Examples/logistic

log using logistic_example.log, text replace
// logistic regression low birth weight exmaple
set seed 3425
webuse lbw
// create center variables
gen center = runiformint(1,3)
tab center, gen(center)

// Fit model in Stata
logit low age smoke center2 center3
estimates store logit


// Now using mlad - write likelhood file in Python
// The dataset is first split into 3 parts to 
// represent 3 centers using setup.py
// I use nojit as this helps when developing
gen cons = 1
mlad  (xb: low  = age smoke center2 center3), ///
      llfile(logistic)                        ///
      pysetup(setup)                          ///
      othervars(low) othervarnames(y)         ///
      nojit
ml display
estimates store mlad 

// compare estimates and standard errors from Stata and Python
estimates table logit mlad, stats(ll) se eq(1)

// some useful mlad options
//   -- nojit - do not jit compile - get better error messages when developing
//   -- mlmethod(d0) - Python just gives likelihood will use numerical derivatives
//   -- mlmethod(d1) - Python just gives likelihood and gradient 
//                     Hessian calculated numerically
//   -- mlmethod(d1debug) gradient - compares gradient from Python with numeric derivatives
//   -- mlmethod(d2debug) hessian  - compares hessian from Python with numeric derivatives
//   -- most ml options will work, e.g for initial values / tolerance etc
log close

