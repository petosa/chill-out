import time
from train import PolicyEvaluator
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy
import torch
import alexnet

policy = get_gradual_unfreezing_policy()
model = alexnet.alexnet(pretrained=True)
starting_model_name = 'models/' + str(time.time()) + '.pt'
torch.save(model.state_dict(), starting_model_name)
evaluator = PolicyEvaluator(model_class=alexnet.AlexNet, verbose=True)

child_filename = starting_model_name
for step in policy:
    child_filename, accuracy = evaluator.train(child_filename, step)
    #Store filename, policystep + state, accuracy


