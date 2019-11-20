import numpy as np
from searcher.env.env import Environment
from itertools import permutations
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import util


class FreezerEnv(Environment):


    # We want to call train on all of the children in get_children and cache the values and return the cached value in evaluate()
    def __init__(self, model, trainer, smell_size, mode, max_depth):
    
        # Validate mode
        modes = ["toggle", "full"]
        if mode not in modes:
            raise Exception("Mode must be one of " + str(modes))
        
        # Class variables
        self.model = model
        self.trainer = trainer
        self.smell_size = smell_size
        self.mode = mode
        self.max_depth = max_depth
        self.model_id = 0
        self.cache = {0: np.inf}

        # Define initial state
        num_layers = len(util.get_trainable_layers(self.model))
        initial_state = (tuple([False] * num_layers), 0, 0, 0) # Initial state is all frozen and with root id (0).
        super().__init__(initial_state)

        # Save initial model to disk
        util.full_save(self.model, self.model_id, self.trainer.session)

    
    def get_train_and_smell_loaders(self):
        train_idx = self.trainer.train_loader.sampler.indices
        np.random.shuffle(train_idx)
        train_idx, smell_idx = train_idx[self.smell_size:], train_idx[:self.smell_size]
        print("Train_idx", len(train_idx), "smell_idx", len(smell_idx))
        train_sampler, smell_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(smell_idx)
        dataset, batch_size = self.trainer.train_loader.dataset, self.trainer.train_loader.batch_size
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
        smell_loader = DataLoader(dataset, batch_size=batch_size, sampler=smell_sampler, num_workers=4, pin_memory=True)
        return train_loader, smell_loader
        

    # Returned children depend on mode.
    def get_children(self, state):
        
        if state[3] >= self.max_depth: # Prune
            return []
        
        # Full mode: can set any number of layers.
        if self.mode == "full":
            n = len(state[0])
            templates = [[True]*i + [False]*(n-i) for i in range(n+1)]
            children = []
            for t in templates:
                children += list(set(permutations(t)))
            children = sorted(children)

        # Toggle mode: can only turn on/off one layer.
        elif self.mode == "toggle":
            children = [state[0][0:i] + tuple([not state[0][i]]) + state[0][i+1:] for i in range(len(state[0]))]
            children.append(state[0][:])
        
        children = [c for c in children if any(c)] # Filter out any "all false" states, these would cause a training error (no trainable weights).
        children = [(c, self.model_id+i+1, state[1], state[3]+1) for i, c in enumerate(children)] # Add current id, parent id, and depth to state.
        self.model_id = children[-1][1] # Update global model id to be the latest assigned id.
        return children
        

    # Evaluate & cache state, or simply restore state evaluation from cache.
    def evaluate(self, state):
        if state[1] not in self.cache:
            train_loader, smell_loader = self.get_train_and_smell_loaders()
            self.cache[state[1]] = self.trainer.train(model=self.model, source=state[2], destination=state[1], freeze_state=state[0],
                train_loader=train_loader, val_loader=smell_loader, test_loader=self.trainer.val_loader)
        return self.cache[state[1]]

        
