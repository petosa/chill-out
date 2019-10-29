import torch
import time
import sys
import numpy as np

from train import *
from searcher.env.env import Environment
from util import *
import torchvision.models as models

class PolicyEnv(Environment):
    # We want to call train on all of the children in get_children and cache the values and return the cached value in evaluate()
    def __init__(self, model_class="alexnet", num_layers=8, verbose=True):
        # config, current model id, parent model id
        initial_state = (tuple([False] * num_layers), 0, 0)
        super().__init__(initial_state)

        self.model_id = 0
        
        session = int(time.time())
        self.trainer = Trainer(session, None, verbose=True, batch_size=256)
        if model_class == "alexnet":
            self.model = load_alexnet(10)
        self.optimizer = torch.optim.SGD(self.model.parameters(), 1e-3, momentum=0.9, nesterov=True, weight_decay=1e-4)
        full_save(self.model, self.optimizer, self.model_id, session)
        #root_loss = self.trainer.train(self.model, self.optimizer, 0, 0, state[0])
        self.cache = {0: 99999999}
        #trainable_layers = util.get_trainable_layers(model)
        #print(len(trainable_layers))

      

    def get_children(self, state):
        parent_loss = self.cache[state[2]]
        #current_loss = self.trainer.train(self.model, self.optimizer, state[2], state[1], state[0])
        is_root = False
        if state[1] == state[2]:
            is_root = True
        if not is_root:
            current_loss = self.evaluate(state)

        if not is_root and current_loss > parent_loss :
            return []
        else:
            children = []
            for i in range(len(state[0])):
                child = (state[0][0:i] + tuple([not state[0][i]]) + state[0][i+1:], self.model_id + i + 1, state[1])
                if any(child[0]):
                    children.append(child)
            children.append(((state[0][:], self.model_id + len(children) + 1, state[1])))
            #print(children)
            self.model_id = self.model_id + len(children)
            #print(self.model_id)
            return children
        
    def evaluate(self, state):
        if state[1] not in self.cache:
            #self.cache[state[1]] = self.trainer.train(self.model, self.optimizer, state[2], state[1], state[0])
            self.cache[state[1]] = self.cache[state[2]]-(np.random.rand()*2-1)
        return self.cache[state[1]]

        
