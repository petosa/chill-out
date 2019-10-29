import torch
import time
import sys
import numpy as np

from train import *
from searcher.env.env import Environment
from util import *

class PolicyEnv(Environment):
    # We want to call train on all of the children in get_children and cache the values and return the cached value in evaluate()
    def __init__(self, model_class="alexnet", verbose=True):

        # Model stuff
        # Model, optimizer, cache
        session = int(time.time())
        self.trainer = Trainer(session, None, verbose=True, batch_size=256)
        if model_class == "alexnet":
            self.model = load_alexnet(10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 1e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)
        
        
        # Search stuff
        # config, current model id, parent model id
        num_layers = len(util.get_trainable_layers(self.model))
        initial_state = (tuple([False] * num_layers), 0, 0)
        super().__init__(initial_state)
        self.model_id = 0
        self.cache = {0: 99999999}
        full_save(self.model, self.optimizer, self.model_id, session)
        

    def get_children(self, state):
        
        parent_loss = self.cache[state[2]]
        
        is_root = state[1] == 0
        if not is_root:
            current_loss = self.evaluate(state)
            if current_loss >= parent_loss:
                return [] # Prune
        
        children = [(state[0][0:i] + tuple([not state[0][i]]) + state[0][i+1:], self.model_id + i + 1, state[1]) for i in range(len(state[0]))]
        children.append(((state[0][:], self.model_id + children[-1][1] + 1, state[1])))
        children = [c for c in children if any(c[0])]
        self.model_id = children[-1][1]
        return children
        

    def evaluate(self, state):
        if state[1] not in self.cache:
            self.cache[state[1]] = self.trainer.train(self.model, self.optimizer, state[2], state[1], state[0])
            #self.cache[state[1]] = self.cache[state[2]]-(np.random.rand()*2-1)
        return self.cache[state[1]]

        
