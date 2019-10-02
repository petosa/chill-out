from train import PolicyEvaluator
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy
import alexnet

policy = get_gradual_unfreezing_policy()
model = alexnet.alexnet(pretrained=True)
evaluator = PolicyEvaluator(model=model)

for step in policy:
    #parent_model = evaluator.model
    #child_model = type(parent_model)() # get a new instance
    #child_model.load_state_dict(parent_model.state_dict())
    accuracy = evaluator.train(step)


