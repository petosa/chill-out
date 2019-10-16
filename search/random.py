from .search import Search
from collections import deque
from random import Random

# TODO
# Write tests
# Make none=infinite rollouts
class RandomRollout(Search):

    def __init__(self, env, max_depth=10000, seed=0):
        self.R = Random(seed)
        super().__init__(env)
        self.max_depth = max_depth
        self.state = (self.env.initial_state, 0)

    def next(self):
        parent, d = self.state
        neighbors = self.env.get_children(parent)
        if d > self.max_depth or len(neighbors) == 0:
            self.state = (self.env.initial_state, 0)
        else:
            curr = self.R.sample(neighbors, 1)[0]
            self.hook(curr, parent)
            self.state = (curr, d + 1)
        return parent, self.env.evaluate(parent)






