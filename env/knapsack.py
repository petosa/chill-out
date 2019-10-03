from .env import Environment
from random import Random

class Knapsack(Environment):

    def __init__(self, weight_limit=15, items=30, max_weight=10, max_value=10, seed=0, verbose=True):
        R = Random(seed)
        initial_state = tuple([])
        super().__init__(initial_state)
        # Value, weight tuples
        self.items = [(R.random()*max_value, R.random()*max_weight) for _ in range(items)]
        if verbose:
            avg_value = round(sum([i[0] for i in self.items])/len(self.items), 4)
            avg_weight = round(sum([i[1] for i in self.items])/len(self.items), 4)
            print("Average value: {}, average weight: {}".format(avg_value, avg_weight))
        self.limit = weight_limit

    def get_children(self, state):
        current_weight = sum([i[1] for i in state])
        neighbors = [tuple(sorted(list(state) + [i])) for i in self.items if i not in state and current_weight + i[1] < self.limit]
        return neighbors
        
    def evaluate(self, state):
        return -sum([i[0] for i in state])
