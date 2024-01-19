# 0. Create some data (saved to disk in a csv)
# 1. run mlad until convergence 
# 2. run mlad and for each iteration in python
#    - output the current parameter estimates 
#    - start the optmizer again
#    - input the current parameter estimates 
import pandas as pd 
import stata_setup

# initialize stata env within python 
stata_setup.config('/Applications/Stata/', 'mp', splash=False)

from pystata import config, stata
# write stata output to a log file
config.set_output_file(filename='stepwise_model.log', replace=True)

for _ in range(3):
     
     # TODO: re-use previous estimates for initalization 
     
     # run stata        
     cmd = """insheet using "data.csv", comma clear

     // declare the data in memory to be survival data
     stset t, failure(d=1)

     // fixing the number of steps in the optimizer to a singe 
     // iteration 
     set maxiter 0
     
     gen beta0 = uniform()

     mlad (ln_lambda: = x1 x2)   ///
          (ln_gamma: ),          /// 
           init(beta0)                ///
           othervars(_t _d)      ///
           othervarnames(t d)    ///
           llfile(weibull_ll)  
     """
     stata.run(cmd)
          
# close the log file 
config.close_output_file()