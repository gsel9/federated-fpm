import stata_setup
# initialize stata env within python 
stata_setup.config('/Applications/Stata/', 'mp', splash=False)

from pystata import stata
# run stata command
cmd = """use http://www.stata-press.com/data/r18/iris, clear"""
stata.run(cmd)

import numpy as np 
from sfi import Data
#X = np.array(Data.get("seplen sepwid petlen petwid"))
#y = np.array(Data.get("iris"))
#print(y)
