class Search():

    # Graph is a function mapping a state to its neighbors.
    def __init__(self, env):
        self.graph = env.graph
        self.evaluate = env.evaluate
        self.frontier = [env.initial_state]
        self.visited = set(self.frontier)


    def visit(self, state):
        raise NotImplementedException
