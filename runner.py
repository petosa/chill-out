import json
from strategy_search import run_search
from baselines import run_baseline

for training_data_amt in [1000]:

    config = json.load(open("config.json", "r"))
    config["trainer"]["args"]["seed"] = 321
    json.dump(config, open("config.json", "w"), indent=4)

    run_search()
    #run_baseline("uf")
    #run_baseline("gu")
    #run_baseline("ct")
