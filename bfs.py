from search import Search
from collections import deque

class BFS(Search):

    def __init__(self, env):
        super().__init__(env)
        self.frontier = deque()
        self.frontier.append(env.initial_state)
        
    def next(self):
        if len(self.frontier) > 0:
            state = self.frontier.pop()
            neighbors = self.graph(state)
            self.frontier += neighbors
            return state, self.evaluate(state)






