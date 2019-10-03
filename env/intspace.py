from .env import Environment

class IntSpace(Environment):

    def __init__(self, goal=-12):
        initial_state = 0
        super().__init__(initial_state)
        self.goal = goal

    def graph(self, state):
        return [x for x in [state-10, state-7, state+7, state+10] if abs(x) < 5000]
        
    def evaluate(self, state):
        return abs(state - self.goal)
