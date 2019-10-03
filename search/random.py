from .search import Search
from collections import deque
from random import Random


class RandomRollout(Search):

    def __init__(self, env, depth=10, seed=0):
        self.R = Random(seed)
        super().__init__(env)
        self.depth = depth
        self.state = (self.env.initial_state, 1)

    def next(self):
        parent, d = self.state
        neighbors = self.env.get_children(parent)
        if d >= self.depth or len(neighbors) == 0:
            self.state = (self.env.initial_state, 1)
            return self.next()
        else:
            curr = self.R.sample(neighbors, 1)[0]
            self.hook(curr, parent)
            self.state = (curr, d + 1)
            return curr, self.env.evaluate(curr)






