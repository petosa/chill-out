from env import Environment

class IntSpace(Environment):

    def __init__(self, initial_state, goal):
        super().__init__(initial_state)
        self.goal = goal
        self.visited = set([initial_state])
        self.history = []

    def graph(self, state):
        all_neighbors = [state-10, state-7, state+7, state+10]
        neighbors = set([n for n in all_neighbors if n not in self.visited])
        self.visited.update(neighbors)
        self.history.extend([(n, state) for n in neighbors])
        return list(neighbors)
        
    def evaluate(self, state):
        return abs(state - self.goal)
