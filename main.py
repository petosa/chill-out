from bfs import BFS
from visitbfs import VisitBFS
from intspace import IntSpace
from knapsack import Knapsack

env = IntSpace()   
search = VisitBFS(env)

s, v = None, None
iters = 0
for _ in range(1000000):
    if iters % 5000 == 0:
        print(iters, len(search.frontier))
    iters += 1
    pair = search.next()
    if pair is None: break
    state, val = pair
    if s is None or val < v:
        s, v = state, val

print(iters, "iter", s, "state", v, "val")
for t in search.trace(s):
    print(t, "\n")
