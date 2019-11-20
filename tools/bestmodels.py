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
    my_best_val, my_best_test = np.inf, np.inf
    val_trace, test_trace = None, None
    for l in f:
        if l.startswith("Final Val Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            if v < my_best_val:
                val_trace = None
                my_best_val = v
        elif l.startswith("Final Test Loss:"):
            v = float(l.replace(",","").split(" ")[3])
            if v < my_best_test:
                test_trace = None
                my_best_test = v
        elif l.startswith("Trace:"):
            trace = eval(l[7:].replace("->",","))
            if val_trace is None:
                val_trace = trace
            if test_trace is None:
                test_trace = trace

print("Val", my_best_val)
[print(step) for step in val_trace]
print("\nTest", my_best_test)
[print(step) for step in test_trace]
