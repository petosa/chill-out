from .search import Search
from collections import deque
from random import sample


class Random(Search):

    def __init__(self, env, depth=10):
        super().__init__(env)
        self.depth = depth

    def next(self):
        curr = self.env.initial_state
        for _ in range(self.depth):
            parent = curr
            neighbors = self.env.graph(parent)
            curr = sample(neighbors, 1)[0]
            self.hook(curr, parent)
        return curr, self.env.evaluate(curr)






