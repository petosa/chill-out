from .env import Environment
from random import Random

class GuessMyNumbers(Environment):

    def __init__(self, limit=1000, bag_size=250, noise=5, seed=0):
        self.noise = noise
        self.R = Random(seed)
        initial_state = tuple()
        super().__init__(initial_state)
        self.limit = limit
        self.bag = set([self.R.randint(0, limit) for _ in range(bag_size)])

    def get_children(self, state):
        return [tuple(sorted([x] + list(state))) for x in range(self.limit) if x not in state and len(state) < len(self.bag)]
        
    def evaluate(self, state):
        return -sum([1 for x in state if x in self.bag]) + sum(1 for x in state if x not in self.bag) + self.R.randint(-self.noise,self.noise)
