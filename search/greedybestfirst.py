from search.search import Search
import heapq as pq

class GreedyBestFirst(Search):

    def __init__(self, env, visited_set=True):
        super().__init__(env)
        self.frontier = []
        self.buffer = []
        pq.heappush(self.frontier, (None, env.initial_state))
        self.visited_set = visited_set
        self.visited = set([env.initial_state])

    def next(self):
        if len(self.buffer) > 0:
            state = self.buffer[0]
            del self.buffer[0]
            value = self.env.evaluate(state)
            pq.heappush(self.frontier, (value, state))
            return state, value
            
        elif len(self.frontier) > 0:
            state = pq.heappop(self.frontier)[1]
            neighbors = self.env.get_children(state)
            if self.visited_set:
                neighbors = [n for n in neighbors if n not in self.visited]
                self.visited.update(set(neighbors))
            self.buffer = neighbors
            [self.hook(n, state) for n in neighbors]
            return self.next()
            






