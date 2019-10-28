import torch
import util
import time
import sys

from train import Trainer
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy


session = int(time.time())
trainer = Trainer(session, None, verbose=True, batch_size=256)

model  = util.load_alexnet(10)
trainable_layers = util.get_trainable_layers(model)
print(len(trainable_layers))
#optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-3)
optimizer = torch.optim.SGD(model.parameters(), 1e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)

modelid = 0
util.full_save(model, optimizer, modelid, session)


pol = sys.argv[1]
print(pol)

if pol == "gu":
    policy = get_gradual_unfreezing_policy(n_layers=len(trainable_layers))
elif pol == "ct":
    policy = get_chain_thaw_policy()
elif pol == "uf":
    policy = [[True]*8]*10
else:
    quit()
 
for step in policy:
    loss = trainer.train(model, optimizer, modelid, modelid+1, step)
    modelid += 1

