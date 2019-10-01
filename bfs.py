from search import Search

class BFS(Search):

    def __init__(self, env):
        super().__init__(env)
        
    def visit(self):
        if len(self.frontier) > 0:
            state = self.frontier[0]
            del self.frontier[0]
            self.visited.add(state)
            neighbors = [n for n in self.graph(state) if n not in self.visited]
            self.frontier += neighbors
            return state, self.evaluate(state)




