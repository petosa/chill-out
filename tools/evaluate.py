'''
Given a session and config, compute the error for each split of that session's config.
'''

# Add cwd to script path
import sys
import os
sys.path.append(os.getcwd())

import util

session, ckpt = sys.argv[1].split("/")
ckpt = int(ckpt.split(".")[0])

config = util.load_config(os.path.join(session, "config.json"))
t = util.make_trainer(config, session)
m = util.make_model(config)
util.full_load(m,ckpt,session)

print("Network Train:", t.evaluate(m, "network_train"))
print("Search Train:", t.evaluate(m, "search_train"))
print("Val:", t.evaluate(m, "val"))
print("Test:", t.evaluate(m, "test"))
