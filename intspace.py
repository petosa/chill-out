from env import Environment

class IntSpace:

    def __init__(self, initial_state, goal):
        self.initial_state = initial_state
        self.goal = goal

    def graph(self, state):
        return [state-10, state-7, state+7, state+10]
        
    def evaluate(self, state):
        return abs(state - self.goal)
