class Search():

    # get_children is a function mapping a state to its neighbors.
    def __init__(self, env):
        self.env = env
        self.history = set()

    def next(self):
        raise NotImplementedError

    def hook(self, state, parent):
        self.history.add((state, parent))

    def trace(self, state):
        curr, trace = state, [state]
        while curr is not self.env.initial_state:
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
