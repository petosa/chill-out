import util
import time
import sys

from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy



def run_baseline(pol):
    
    modelid = 0
    session = int(time.time())
    trainer = util.make_trainer(session)
    model = util.make_model()
    trainable_layers = util.get_trainable_layers(model)
    util.full_save(model, modelid, session)

    # Validate input
    if pol == "gu":
        policy = get_gradual_unfreezing_policy(n_layers=len(trainable_layers))
    elif pol == "ct":
        policy = get_chain_thaw_policy()
    elif pol == "uf":
        policy = [[True]*8]*10
    elif pol == "custom":
        policy = [[False, False, True, False, False, False, False, True]]#, [True, False, False, False, False, False, True, True]]
    else:
        quit()

    # Bookkeeping
    util.copy_config(session, additional_info={
        "baseline": pol,
        "policy": policy
    })

    # Train loop
    for step in policy:
        loss = trainer.train(model, modelid, modelid+1, step)
        modelid += 1


if __name__ == "__main__":
    pol = sys.argv[1]
    run_baseline(pol)