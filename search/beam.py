from search.search import Search
import heapq as pq

class Beam(Search):

    def __init__(self, env, beam_size=3, visited_set=True):
        super().__init__(env)
        self.frontier = []
        self.buffer = [env.initial_state]
        self.beam_size = beam_size
        self.visited_set = visited_set
        self.visited = set([env.initial_state])
        self.evaluating = False

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
            self.frontier = self.frontier[:self.beam_size]
            print("beam", self.frontier)
            for _ in self.frontier:
                state = pq.heappop(self.frontier)[1]
                neighbors = self.env.get_children(state)
                if self.visited_set:
                    neighbors = [n for n in neighbors if n not in self.visited]
                    self.visited.update(set(neighbors))
                self.buffer += neighbors
                [self.hook(n, state) for n in neighbors]
            return self.next()







