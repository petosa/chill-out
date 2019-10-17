from .search import Search
from collections import deque
from random import Random


# Note: Does not memoize previously visited states. The reason is we have no way of telling once all states are expanded.
# Feel free to memoize states on the environment side if your evaluation is expensive.
# TODO More tests
class RandomRollout(Search):

    def __init__(self, env, max_depth=None, seed=0):
        self.R = Random(seed)
        super().__init__(env)
        self.max_depth = max_depth
        self.state = (self.env.initial_state, 0)
        self.visited = set([env.initial_state])

    def next(self):

        parent, d = self.state
        neighbors = self.env.get_children(parent)
        neighbors = [n for n in neighbors if n not in self.visited]
    
        if (self.max_depth is not None and d >= self.max_depth) or len(neighbors) == 0:
            self.state = (self.env.initial_state, 0)
            self.visited = set([self.env.initial_state])
        else:
            curr = self.R.sample(neighbors, 1)[0]
            self.visited.add(curr)
            self.hook(curr, parent)
            self.state = (curr, d + 1)

        return parent, self.env.evaluate(parent)






