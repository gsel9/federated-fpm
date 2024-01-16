""" Do in terminal 

>>> python -m pip install pystata stata-setup

The pystata Python package is shipped with Stata and located 
in the pystata subdirectory of the utilities folder in Stata’s 
installation directory. You can type the following in Stata to 
view the name of this directory:

   . display c(sysdir_stata)

Add to .zshrc (alternatively: ~/.bashrc or ~/.bash_profile)
    
    export PYTHONPATH="/Applications/Stata/utilities:$PYTHONPATH"
    
"""


import stata_setup
# initialize stata env within python 
stata_setup.config('/Applications/Stata/', 'mp', splash=False)

from pystata import config, stata
# write stata output to a log file
config.set_output_file(filename='output_stata_from_py.log', replace=True)

# run stata command
cmd = """use "/Users/sela/Desktop/stata-data/example_data.dta", clear
stpm2 sex rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3  if site23 == 4, scale(hazard) bhazard(rate) iterate(50)  df(4) tvc(rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3) dftvc(2)"""
stata.run(cmd)

# close the log file 
config.close_output_file()
