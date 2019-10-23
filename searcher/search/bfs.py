from .search import Search
from collections import deque


# TODO More tests
class BFS(Search):

    def __init__(self, env, sample=None, seed=0):
        super().__init__(env, sample, seed)
        self.frontier = deque()
        self.frontier.append(env.initial_state)
        self.visited = set([env.initial_state])

    def next(self):
        if len(self.frontier) > 0:
            state = self.frontier.popleft()
            neighbors = self.env.get_children(state)
            neighbors = [n for n in neighbors if n not in self.visited]
            neighbors = self.sample_list(neighbors)
            self.visited.update(set(neighbors))
            self.frontier += neighbors
            [self.hook(n, state) for n in neighbors]
            return state, self.env.evaluate(state)






