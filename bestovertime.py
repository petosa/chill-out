import sys
import numpy as np

session = sys.argv[1]
logfile = session + "/log.txt"


with open(logfile, "r") as f:
    my_best = np.inf
    for l in f:
        if "Iteration" in l:
            print(my_best)
        elif len(l.split(","))==5:
            v = float(l.split(",")[3])
            my_best = min(my_best,v)


            
