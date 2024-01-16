// embed python code between a python-end block 
python

print('Hello, Python!')

import numpy as np

x = np.arange(5)
print(x)

// terminate the python session 
end 

// TODO: Execute a Python script file (see `python help` in stata)
// python script pyfilename [, args(args_list) global
//                userpaths(user_paths[, prepend])]
