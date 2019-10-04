from env.env import Environment
from train import PolicyEvaluator
import time
import alexnet
import torch

class PolicyEnv(Environment):

    def __init__(self, model_class=alexnet.alexnet, num_layers=8, verbose=True):
        initial_state = (tuple([False] * num_layers), 0, 0)
        super().__init__(initial_state)

        model = model_class(pretrained=True)
        self.model_id = 0
        self.starting_model_name = 'models/0.pt'
        self.evaluator = PolicyEvaluator(epochs=1, model_class=model_class, verbose=True)
        torch.save(model.state_dict(), self.evaluator.save_dir + self.starting_model_name)
        self.parent_losses = {}

        

    def get_children(self, state):
        children = [(state[0][0:i] + tuple([not state[0][i]]) + state[0][i+1:], self.model_id + i + 1, state[1]) for i in range(len(state[0]))]
        children.append(((state[0][:], self.model_id + len(children) + 1, state[1])))
        children.append((None, state[1], state[1]))
        new_children = []
        for child in children:
            if child[0] == None:
                continue
            if any(child[0]):
                new_children.append(child)
        self.model_id = self.model_id + len(children)
        return children
        
    def evaluate(self, state):
        if state[0] == None:
            return self.parent_losses[state[1]]
        if not any(state[0]):
            return 99999999
        loss = self.evaluator.train('models/' + str(state[2]) + '.pt', 'models/' + str(state[1]) + '.pt', state[0])
        self.parent_losses[state[2]] = loss
        return loss
        
