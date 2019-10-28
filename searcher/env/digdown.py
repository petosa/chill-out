from .env import Environment
from random import Random
from collections import defaultdict

class DigDown(Environment):

    def __init__(self, branching_factor=30, max_depth=100, seed=0):
        super().__init__((0,0))
        self.tree = defaultdict(list)
        self.R = Random(seed)
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        
    def get_children(self, state):
        if state not in self.tree:
            for _ in range(self.branching_factor):
                child = (state[0] - self.R.random(), state[1]+1)
                if self.R.random() < self.R.random() and child[1] <= self.max_depth:
                    self.tree[state].append(child)
        return self.tree[state]

        
    def evaluate(self, state):
        return state[0]
