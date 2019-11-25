import json
from freezersearch import run_search
from searcher.search.beam import Beam
from fixedpolicy import run_policy
import util
import sys
import os
import time

config = util.load_config("config.json")

if sys.argv[1] == "search":
    # Hyperparameters
    mode = "toggle"
    max_depth = None
    halve = True
    search_algo = Beam
    search_kwargs = {
        "beam_size": 10,
        "sample": None
    }
    run_search(config, mode, max_depth, halve, search_algo, search_kwargs)

# Train under a fixed policy
elif sys.argv[1] == "policy":

    model = util.make_model(config)    
    n_layers = len(util.get_trainable_layers(model))
    if sys.argv[2] == "gu": # Gradual unfreezing
        p = util.get_gradual_unfreezing_policy(n_layers)
    elif sys.argv[2] == "ct": # Chain thaw
        p = util.get_chain_thaw_policy(n_layers)
    elif sys.argv[2] == "uf": # Unfrozen
        p = [[True]*n_layers]*8
    else: # Replay
        splitted = sys.argv[2].split("/")
        session, ckpt = splitted[:-1], splitted[-1]
        with open(os.path.join("/".join(session), "log.txt"), "r") as f:
            remember = False
            for l in f:
                if l.startswith("Model saved") and ("/" + ckpt) in l:
                    remember = True
                elif remember and l.startswith("Trace:"):
                    p = [list(x[0]) for x in eval(l[7:].replace("->",","))]
                    break
        p = p[1:]

    run_policy(p)

