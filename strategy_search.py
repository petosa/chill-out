from searcher.search.bestfirst import BestFirst
from searcher.search.beam import Beam
from freezer import Freezer
import util
import time


def run_search():
    # Hyperparameters
    mode = "toggle"
    search_algo = Beam
    search_args = {
        "beam_size": 6,
        "sample": None
    }

    # Initialize
    session = int(time.time())
    trainer = util.make_trainer(session)
    model = util.make_model()
    env = Freezer(model, trainer, mode=mode)
    search = search_algo(env, **search_args)

    # Bookkeeping
    util.copy_config(session, additional_info={
        "search": {
            "name": search_algo.__name__,
            "mode": mode,
            "args": search_args
        }
    })

    # Search loop
    s, v = None, None
    iters = 0
    while True:
        iters += 1
        pair = search.next()
        if pair is None: break # No more nodes to expand
        state, val = pair
        if s is None or val < v:
            s, v = state, val
        env.trainer.log_line("*"*20)
        env.trainer.log_line("Iteration {}. Best error: {}".format(iters, v))
        env.trainer.log_line("->".join([str(s) for s in search.trace(s)]))
        env.trainer.log_line("*"*20)
    env.trainer.log_line("Completed all tasks.")



if __name__ == "__main__":
    run_search()
