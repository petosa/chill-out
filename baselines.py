import time
from train import PolicyEvaluator
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy
import torch
import alexnet
import squeezenet
import util

model = alexnet.alexnet(pretrained=True)
#model = squeezenet.squeezenet1_1(pretrained=True)
num_trainable = util.get_trainable_layer_count(model)
#policy = [[True]*num_trainable]
policy = get_gradual_unfreezing_policy()
# policy = get_chain_thaw_policy()

modelid = 0
starting_model_name = 'models/' + str(modelid) + '.pt'
log_file_name = 'logs/' + str(modelid) + '.txt'
print ("Log File: {}".format(log_file_name))
evaluator = PolicyEvaluator(batch_size=256, model_class=type(model), verbose=True, epochs=80, log_file=log_file_name)
torch.save(model.state_dict(), evaluator.save_dir + starting_model_name)

child_filename = starting_model_name
for step in policy:
    dest_filename = 'models/' + str(modelid+1) + '.pt'
    loss = evaluator.train(child_filename, dest_filename, step)
    modelid += 1
    child_filename = 'models/' + str(modelid) + '.pt'
    #Store filename, policystep + state, accuracy


