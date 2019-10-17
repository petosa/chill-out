import sys
sys.path.append("..")
import search
from env.knapsack import Knapsack
from env.intspace import IntSpace
from env.maze import Maze

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
    #e = Knapsack(items=25, weight_limit=25)
    e = Maze(60, 60, 1)
    all_visited = []
    se = s(e)
    current_best = None
    for i in range(1000):
        node = se.next()
        if node is None: break
        state,val = node
        current_best = val if current_best is None else min(current_best,val)
        all_visited.append(state)
        log[s].append(-current_best)
    e.visualize(all_visited)


for s in searches:
    l = plt.plot(log[s], linewidth=5, label=str(s.__name__) + ": " + str(round(log[s][-1], 1)), alpha=0.9)
plt.title("Search algorithm comparison on knapsack environment", fontsize=15)
plt.ylabel("Best knapsack value found", fontsize=15)
plt.xlabel("Nodes expanded", fontsize=15)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()




