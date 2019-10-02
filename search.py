class Search():

    # Graph is a function mapping a state to its neighbors.
    def __init__(self, env):
        self.graph = env.graph
        self.evaluate = env.evaluate
        self.initial_state = env.initial_state
        self.history = env.history

    def next(self):
        raise NotImplementedException

    def trace(self, state):
        curr, trace = state, [state]
        while curr is not self.initial_state:
            found = False
            for pair in self.history:
                if pair[0] is curr:
                    curr = pair[1]
                    found = True
                    break
            if not found:
                return None
            trace.append(curr)
        return trace[::-1]
