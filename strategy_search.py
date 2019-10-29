from searcher.search.bestfirst import BestFirst
from searcher.search.beam import Beam
from policyenv import PolicyEnv
import util

env = PolicyEnv()
search = BestFirst(env)
search = Beam(env, beam_size=3)

s, v = None, None
iters = 0
for _ in range(1000000):
    iters += 1
    pair = search.next()
    if pair is None: break
    state, val = pair
    if s is None or val < v:
        s, v = state, val
    env.trainer.log_line("*"*20)
    env.trainer.log_line("Iteration {}. Best error: {}".format(iters, v))
    env.trainer.log_line("->".join([str(s) for s in search.trace(s)]))
    env.trainer.log_line("*"*20)
env.trainer.log_line("Completed all tasks.")

