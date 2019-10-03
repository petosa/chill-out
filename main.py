from search.bfs import BFS
from search.random import RandomRollout
from search.greedybestfirst import GreedyBestFirst

from env.intspace import IntSpace
from env.knapsack import Knapsack

env = Knapsack()   
search = BFS(env, visited_set=False)

s, v = None, None
iters = 0
for i in range(3000):
    if i % 10000 == 0:
        print("Step", i)
    iters += 1
    pair = search.next()
    if pair is None: break
    state, val = pair
    if s is None or val < v:
        s, v = state, val

print(iters, "iter", s, "state", v, "val")
print("Trace:")
for t in search.trace(s):
    print(t)
