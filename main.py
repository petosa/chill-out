from search.bfs import BFS
from search.visitbfs import VisitBFS
from search.random import Random
from env.intspace import IntSpace
from env.knapsack import Knapsack

env = IntSpace()   
search = Random(env)

s, v = None, None
iters = 0
for i in range(150):
    if i % 10000 == 0:
        print("Step", i)
    iters += 1
    pair = search.next()
    if pair is None: break
    state, val = pair
    if s is None or val < v:
        s, v = state, val

print(iters, "iter", s, "state", v, "val")
print(s)
for t in search.trace(s):
    print(t, "\n")
