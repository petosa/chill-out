from .search import Search


# TODO More tests
class Greedy(Search):

    def __init__(self, env, sample=None, seed=0):
        super().__init__(env, sample, seed)
        self.best = None
        self.buffer = [env.initial_state]
        self.visited = set([env.initial_state])

    def next(self):

        # First attempt to evaluate any nodes in the buffer
        if len(self.buffer) > 0:
            state = self.buffer[0]
            del self.buffer[0]
            value = self.env.evaluate(state)
            self.best = (value, state) if self.best is None else min((value, state), self.best)
            return state, value

        # Selected node is the best of nodes we just evaluated.
        elif self.best is not None:
            state = self.best[1]
            neighbors = self.env.get_children(state)
            neighbors = [n for n in neighbors if n not in self.visited]
            neighbors = self.sample_list(neighbors)
            self.visited.update(set(neighbors))
            self.buffer += neighbors
            [self.hook(n, state) for n in neighbors]
            self.best = None
            return self.next()







