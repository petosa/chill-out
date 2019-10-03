from collections import defaultdict

class Search():

    # get_children is a function mapping a state to its neighbors.
    def __init__(self, env):
        self.env = env
        self.history = defaultdict(int)

    def next(self):
        raise NotImplementedError

    def hook(self, state, parent):
        if state not in self.history.keys():
            self.history[state] = parent

    def trace(self, state):
        curr, trace = state, [state]
        while curr is not self.env.initial_state:
            if curr not in self.history:
                return
            curr = self.history[curr]
            trace.append(curr)
        return trace[::-1]
