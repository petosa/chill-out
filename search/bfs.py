from .search import Search
from collections import deque

# TODO
# Write tests
class BFS(Search):

    def __init__(self, env, visited_set=True):
        super().__init__(env)
        self.frontier = deque()
        self.frontier.append(env.initial_state)
        self.visited_set = visited_set
        self.visited = set([env.initial_state])

    def next(self):
        if len(self.frontier) > 0:
            state = self.frontier.popleft()
            neighbors = self.env.get_children(state)
            if self.visited_set:
                neighbors = [n for n in neighbors if n not in self.visited]
                self.visited.update(set(neighbors))
            self.frontier += neighbors
            [self.hook(n, state) for n in neighbors]
            return state, self.env.evaluate(state)






