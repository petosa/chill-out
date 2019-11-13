import numpy as np
from searcher.env.env import Environment
from itertools import permutations
import util


class FreezerEnv(Environment):


    # We want to call train on all of the children in get_children and cache the values and return the cached value in evaluate()
    def __init__(self, model, trainer, mode="toggle"):
    
        # Validate mode
        modes = ["toggle", "full"]
        if mode not in modes:
            raise Exception("Mode must be one of " + str(modes))
        
        # Class variables
        self.model = model
        self.trainer = trainer
        self.mode = mode
        self.model_id = 0
        self.cache = {0: np.inf}

        # Define initial state
        num_layers = len(util.get_trainable_layers(self.model))
        initial_state = (tuple([False] * num_layers), 0, 0) # Initial state is all frozen and with root id (0).
        super().__init__(initial_state)

        # Save initial model to disk
        util.full_save(self.model, self.model_id, self.trainer.session)
        

    # Returned children depend on mode.
    def get_children(self, state):
        
        # General (non-root) case; if parent's loss is less than our loss, prune.
        parent_loss = self.cache[state[2]]
        if not state[1] == 0:
            current_loss = self.evaluate(state)
            if current_loss >= parent_loss:
                return [] # Prune
        
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
        children = [(c, self.model_id+i+1, state[1]) for i, c in enumerate(children)] # Add current and parent ids to state.
        self.model_id = children[-1][1] # Update global model id to be the latest assigned id.
        return children
        

    # Evaluate & cache state, or simply restore state evaluation from cache.
    def evaluate(self, state):
        if state[1] not in self.cache:
            self.cache[state[1]] = self.trainer.train(self.model, state[2], state[1], state[0])
        return self.cache[state[1]]

        
