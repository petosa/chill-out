'''
Finds the two models that minimize search-train and validation errors for a session.
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
    st_trace, val_trace = None, None
    for l in f:
        if l.startswith("Final ST Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            if v < my_best_st:
                st_trace = None
                my_best_st = v
        elif l.startswith("Final Val Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            if v < my_best_val:
                val_trace = None
                my_best_val = v
        elif l.startswith("Trace:"):
            trace = eval(l[7:].replace("->",","))
            if st_trace is None:
                st_trace = trace
            if val_trace is None:
                val_trace = trace

print("ST", my_best_st)
[print(step) for step in st_trace]
print("\nVal", my_best_val)
[print(step) for step in val_trace]
