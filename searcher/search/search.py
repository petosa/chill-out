from collections import defaultdict
from random import Random
from math import ceil


class Search():

    # get_children is a function mapping a state to its neighbors.
    def __init__(self, env, sample=None, seed=0):
        self.env = env
        self.sample = sample
        self.R = Random(seed)
        self.history = {env.initial_state: None}
       
    # The first state visited must be the initial state.
    def next(self):
        raise NotImplementedError

    # Connect a state to its parent in the traceback tree; will overwrite existnig parent if new path is shorter.
    def hook(self, state, parent):
        if state not in self.history.keys() or len(self.trace(parent)) < len(self.trace(self.history[state])):
            self.history[state] = parent

    # Walk the traceback tree
    def trace(self, state):
        curr, trace = state, [state]
        if curr not in self.history: return
        while curr is not self.env.initial_state:
            curr = self.history[curr]
            trace.append(curr)
        return trace[::-1]

    # Given a list, induce a random sample using one of 3 different options.
    def sample_list(self, l):
        num = len(l)
        if self.sample is None: return l
        elif self.sample < 1: return self.R.sample(l, ceil(num*self.sample))
        elif self.sample >= 1: return self.R.sample(l, min(num, self.sample))

