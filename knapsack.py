from env import Environment

class Knapsack(Environment):

    def __init__(self, limit):
        initial_state = tuple([])
        super().__init__(initial_state)
        self.visited = set([initial_state])
        self.history = []
        self.items = ([
            (2.1232, 6.0001), (9.098, 5.4322), (0.0901, 3.0904), (5.3694, 4.6595), (9.8987, 5.1491), (9.5859, 5.8888), (6.7434, 6.2964), (5.9113, 9.3563), (9.2535, 3.2823), (1.6343, 8.1583), (3.4434, 7.0431), (6.6751, 1.6638), (7.7529, 9.5329), (5.847, 0.9938), (4.9031, 8.1807), (3.7376, 2.016), (1.322, 0.1689), (5.7007, 4.6636), (9.6793, 5.832), (8.8013, 8.0845), (8.6443, 4.035), (2.1878, 9.4563), (0.6273, 3.0523), (3.6219, 1.7553), (0.6048, 6.6022), (0.3917, 0.4702), (4.6225, 0.0265), (5.9946, 1.4738), (9.2158, 9.9996), (7.9202, 0.575), (5.6854, 6.8484), (7.0796, 2.9906), (6.1011, 3.8901), (5.7225, 6.1226), (4.8644, 1.6277), (5.0571, 1.3692), (3.1149, 8.9993), (1.396, 9.3782), (2.043, 8.6451), (0.9465, 6.839), (6.9975, 4.3984), (4.0267, 6.9139), (6.3952, 8.7259), (1.6025, 2.2811), (1.3929, 4.309), (8.3651, 4.1974), (4.2977, 6.9632), (3.83, 6.979), (1.3146, 1.4478), (4.1379, 1.6155)
        ])
        self.limit = limit

    def graph(self, state):
        current_weight = sum([i[1] for i in state])
        neighbors = [tuple(sorted(list(state) + [i])) for i in self.items if i not in state and current_weight + i[1] < self.limit]
        neighbors = set([n for n in neighbors if n not in self.visited])
        self.visited.update(neighbors)
        self.history.extend([(n, state) for n in neighbors])
        return neighbors
        
    def evaluate(self, state):
        return -sum([i[0] for i in state])
