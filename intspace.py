from env import Environment

class IntSpace(Environment):

    def __init__(self, goal=-12):
        initial_state = 0
        super().__init__(initial_state)
        self.goal = goal

    def get_children(self, state):
        return [state-10, state-7, state+7, state+10]
        
    def evaluate(self, state):
        return abs(state - self.goal)
