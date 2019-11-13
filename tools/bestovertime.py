'''
Given a session, print out the sequence of best validation and search-train errors found over time.
'''

# Add cwd to script path
import sys
import os
sys.path.append(os.getcwd())

import numpy as np

session = sys.argv[1]
logfile = session + "/log.txt"

with open(logfile, "r") as f:
    my_best_st, my_best_val = np.inf, np.inf
    for l in f:
        if "Iteration" in l:
            print(l.split(" ")[1][:-1], my_best_st, my_best_val)
        elif l.startswith("Final ST Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            my_best_st = min(my_best_st,v)
        elif l.startswith("Final Val Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            my_best_val = min(my_best_val,v)


            
