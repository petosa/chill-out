from bfs import BFS
from intspace import IntSpace


env = IntSpace(0, 12)   
search = BFS(env)

print(search.frontier)
for _ in range(10):
    print(search.visit())
    print(search.frontier)
