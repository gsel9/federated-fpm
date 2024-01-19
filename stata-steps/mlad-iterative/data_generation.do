// Simulate some survival data 

set obs 10000
set seed 987123

gen x1 = rnormal()
gen x2 = rnormal()

// Simulation of survival data
survsim t d, dist(weibull) lambda(0.2) gamma(0.8) maxt(5) cov(x1 0.1 x2 0.1)

outsheet t d x1 x2 using data.csv , comma 
