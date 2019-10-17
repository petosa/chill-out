from collections import defaultdict


class Search():

    # get_children is a function mapping a state to its neighbors.
    def __init__(self, env):
        self.env = env
        self.history = {env.initial_state: None}

    # The first state visited must be the initial state.
    def next(self):
        raise NotImplementedError

    def hook(self, state, parent):
        if state not in self.history.keys() or len(self.trace(parent)) < len(self.trace(self.history[state])):
            self.history[state] = parent

    def trace(self, state):
        curr, trace = state, [state]
        if curr not in self.history: return
        while curr is not self.env.initial_state:
            curr = self.history[curr]
            trace.append(curr)
        return trace[::-1]
