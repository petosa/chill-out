import util
import time
import sys


def run_policy(policy):
    
    modelid = 0
    session = int(time.time())
    config = util.load_config("config.json")
    trainer = util.make_trainer(config, session)
    model = util.make_model(config)
    util.full_save(model, modelid, session)

    # Bookkeeping
    util.copy_config(config, session, additional_info={
        "policy": policy
    })

    # Train loop
    trace = [([False]*len(policy[0]), 0, 0)]
    for step in policy:
        trainer.log_line("Trace: " + "->".join([str(s) for s in trace]))
        loss = trainer.train(model, modelid, modelid+1, step, train_loader=trainer.train_loader, val_loader=trainer.val_loader, test_loader=trainer.test_loader)
        trace.append((step, modelid+1, modelid))
        modelid += 1
    trainer.log_line("Trace: " + "->".join([str(s) for s in trace]))

