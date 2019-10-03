class Environment:


    def __init__(self, initial_state):
        self.initial_state = initial_state

    def graph(self, state):
        raise NotImplementedError
        
    def evaluate(self, state):
        raise NotImplementedError
