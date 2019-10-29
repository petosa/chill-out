from searcher.env.knapsack import Knapsack
from searcher.env.intspace import IntSpace
from searcher.env.gmn import GuessMyNumbers
from searcher.env.maze import Maze
from searcher.env.digdown import DigDown
from policyenv import PolicyEnv


from searcher.search.greedy import Greedy
from searcher.search.bfs import BFS
from searcher.search.random import RandomRollout
from searcher.search.bestfirst import BestFirst
from searcher.search.beam import Beam
from collections import defaultdict
import matplotlib.pyplot as plt
from copy import deepcopy



def experiment(searches, env, iter):

    log = defaultdict(list)

    for s, args in searches:
        key = str(s) + str(args)
        e = deepcopy(env)
        all_visited = []
        args["env"] = e
        algo = s(**args)
        current_best = None
        for _ in range(iter):
            node = algo.next()
            if node is None: break
            state,val = node
            current_best = val if current_best is None else min(current_best,val)
            all_visited.append(state)
            #print(state, val, e.cache[state[2]], e.cache[state[2]] > val)
            log[key].append(-current_best)
        #e.visualize(all_visited, title=algo.__class__.__name__ + " " + str({k:v for k,v in args.items() if k != "env"}), delay=0)
        

        #for t in algo.trace(state):
        #    print(t, e.cache[t[1]], "\n")

    searches = sorted(searches, key=lambda s: -len(log[str(s[0]) + str(s[1])]))
    for s, args in searches:
        del args["env"]
        key = str(s) + str(args)
        plt.plot(log[key], linewidth=5, label=str(s.__name__) + " {}".format(args) + ": " + str(round(log[key][-1], 1)), alpha=0.9)
    plt.title("Search algorithm comparison on {} environment".format(e.__class__.__name__), fontsize=15)
    plt.ylabel("Best reward found", fontsize=15)
    plt.xlabel("Nodes expanded", fontsize=15)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":


    seed=3
    searches = [(BestFirst, {"sample":.5, "seed":0})]
    searches += [(Beam, {"sample":.5,"beam_size":i*3, "seed":0}) for i in range(1,5)]
    searches += [(RandomRollout, {})]
    searches += [(Greedy, {})]
    '''
    searches = [
        #(BFS, {"sample":10}), 
        #(BestFirst, {"sample":10}),
        #(RandomRollout, {}),
        (BestFirst, {"sample":10,"seed":seed}),
        (BestFirst, {"sample":5,"seed":seed}),
        (BestFirst, {"sample":20,"seed":seed}),
        (BestFirst, {"sample":30,"seed":seed}),
        (BestFirst, {"sample":1,"seed":seed}),
        (BestFirst, {"sample":40,"seed":seed}),
        (BestFirst, {"sample":50,"seed":seed}),
        #(Greedy, {"sample":10})
    ]
    '''

    env = PolicyEnv()

    experiment(searches, env, 1000)