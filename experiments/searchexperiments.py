import sys
sys.path.append("..")
import search
from env.knapsack import Knapsack
from env.intspace import IntSpace

from search.greedy import Greedy
from search.bfs import BFS
from search.random import RandomRollout
from search.bestfirst import BestFirst
from search.beam import Beam
from collections import defaultdict
import matplotlib.pyplot as plt

searches = [BFS, BestFirst, RandomRollout, Beam, Greedy]
log = defaultdict(list)

for s in searches:
    e = Knapsack(items=25, weight_limit=25)
    try:
        se = s(e, visited_set=True)
    except:
        se = s(e)
    current_best = None
    for i in range(350):
        v = se.next()
        if v is None: break
        v = v[1]
        current_best = v if current_best is None else min(current_best,v)
        log[s].append(-current_best)

for s in searches:
    l = plt.plot(log[s], linewidth=5, label=str(s.__name__) + ": " + str(round(log[s][-1], 1)), alpha=0.9)
plt.title("Search algorithm comparison on knapsack environment", fontsize=15)
plt.ylabel("Best knapsack value found", fontsize=15)
plt.xlabel("Nodes expanded", fontsize=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()




