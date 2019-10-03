import time
from train import PolicyEvaluator
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy
import torch
import alexnet

# policy = get_gradual_unfreezing_policy()
# policy = get_chain_thaw_policy()
policy = [[True]*8]
model = alexnet.alexnet(pretrained=True)
modelid = 0
starting_model_name = 'models/' + modelid + '.pt'
log_file_name = 'logs/' + modelid + '.txt'
torch.save(model.state_dict(), starting_model_name)
evaluator = PolicyEvaluator(model_class=alexnet.AlexNet, verbose=True, epochs=50, log_file=log_file_name)

child_filename = starting_model_name
for step in policy:
    dest_filename = child_filename + 1
    loss = evaluator.train(child_filename, dest_filename, step)
    child_filename += 1
    #Store filename, policystep + state, accuracy


