from freezerenv import FreezerEnv
import util
import time


def run_search(config, mode, max_depth, halve, search_algo, search_kwargs):

    # Initialize
    session = int(time.time())
    
    # Bookkeeping
    util.copy_config(config, session, additional_info={
        "search": {
            "mode": mode,
            "max_depth": max_depth,
            "halve": halve,
            "name": search_algo.__name__,
            "args": search_kwargs
        }
    })

    config["trainer"]["args"]["halve"] = halve
    trainer = util.make_trainer(config, session)
    model = util.make_model(config)
    env = FreezerEnv(model, trainer, mode, max_depth)
    search = search_algo(env, **search_kwargs)

    # Search loop
    s, v = None, None
    iters = 0
    while True:
        iters += 1
        pair = search.next()
        env.trainer.log_line("Beam level: {}".format(search.level))
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

