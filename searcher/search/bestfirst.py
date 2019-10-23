import heapq as pq
from .search import Search


# TODO More tests
class BestFirst(Search):

    def __init__(self, env, sample=None, seed=0):
        super().__init__(env, sample, seed)
        self.sample = sample
        self.frontier = []
        self.buffer = [env.initial_state]
        self.visited = set([env.initial_state])

    def next(self):

        # First attempt to evaluate any nodes in the buffer
        if len(self.buffer) > 0:
            state = self.buffer[0]
            del self.buffer[0]
            value = self.env.evaluate(state)
            pq.heappush(self.frontier, (value, state))
            return state, value

        # If buffer is empty attempt to fill it by popping parent from frontier.
        elif len(self.frontier) > 0:
            state = pq.heappop(self.frontier)[1]
            neighbors = self.env.get_children(state)
            neighbors = [n for n in neighbors if n not in self.visited]
            neighbors = self.sample_list(neighbors)
            self.visited.update(set(neighbors))
            self.buffer = neighbors
            [self.hook(n, state) for n in neighbors]
            return self.next()
            






