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


for split in ["Train", "Val", "Val 2", "Test"]:
    with open(logfile, "r") as f:
        my_best, my_trace = np.inf, None
        for l in f:
            if l.startswith("Final {} Loss:".format(split)):
                splitted = l.replace(",","").split(" ")
                v = float(splitted[splitted.index("Loss:")+1])
                if v < my_best:
                    my_trace = None
                    my_best = v
            elif l.startswith("Trace:"):
                trace = eval(l[7:].replace("->",","))
                my_trace = trace if my_trace is None else my_trace

        print("{}\n".format(split), my_best)
        if my_best != np.inf:
            [print(step) for step in my_trace]
