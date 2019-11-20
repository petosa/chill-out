from freezerenv import FreezerEnv
import util
import time


def run_search(config, smell_size, mode, max_depth, search_algo, search_kwargs):

    # Initialize
    session = int(time.time())
    
    # Bookkeeping
    util.copy_config(config, session, additional_info={
        "search": {
            "smell_size": smell_size,
            "max_depth": max_depth,
            "mode": mode,
            "name": search_algo.__name__,
            "args": search_kwargs
        }
    })

    trainer = util.make_trainer(config, session)
    smell_size = int(smell_size*len(idx)) if smell_size < 1 else smell_size
    model = util.make_model(config)
    env = FreezerEnv(model, trainer, smell_size, mode, max_depth)
    search = search_algo(env, **search_kwargs)

    # Search loop
    s, v = None, None
    iters = 0
    while True:
        iters += 1
        pair = search.next()
        if pair is None: break # No more nodes to expand
        state, val = pair
        env.trainer.log_line("Trace: " + "->".join([str(s) for s in search.trace(state)]))
        if s is None or val < v:
            s, v = state, val
        env.trainer.log_line("*"*20)
        env.trainer.log_line("Iteration {}. Best error: {}".format(iters, v))
        env.trainer.log_line("->".join([str(s) for s in search.trace(s)]))
        env.trainer.log_line("*"*20)
    env.trainer.log_line("Completed all tasks.")

