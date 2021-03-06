from torch import save, load
import numpy as np
import os
import torch
import random
import json
import models
import loaders
from train import Trainer
from copy import deepcopy


# Returns a list of layer objects from the given model.
def get_trainable_layers(model):
    layers = []
    for _, w in model.named_children():
        children = list(w.named_children())
        if len(children) == 0 and len(list(w.named_parameters())) > 0:
            layers.append(w) # Shallow layers
        else:
            for _, w1 in children:
                if len(list(w1.named_parameters())) > 0:
                    layers.append(w1) # Nested layers
    return layers

# Save a model with a given ID to the given session folder.
def full_save(model, id, session):
    try: os.mkdir(str(session))
    except: pass
    data = {"model":model.state_dict()}
    save(data, os.path.join(str(session), str(id) + ".pt"))

# Load a model with a given ID from the given session folder.
def full_load(model, id, session):
    state_dict = load(os.path.join(str(session), str(id) + ".pt"))
    model.load_state_dict(state_dict["model"])

# Class for determining once error has been increasing for too many iterations, remembers last best model weights before divergence.
class EarlyStopping:
    def __init__(self, patience, verbose=False, delta=0):
        self.best_model = None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = np.inf
        self.early_stop = False
        self.delta = delta

    def update(self, val_loss, model):
        if self.best_val_loss > val_loss + self.delta:
            if self.verbose: print(f'Loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).')
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, val_loss, model):
        self.best_model = deepcopy(model.state_dict())

# Seeds all aspects for torch execution.
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Loads run configuration
def load_config(path):
    return json.load(open(path, "r"))

# Saves run configuration
def save_config(config, path):
    json.dump(config, open(path, "w"), indent=4)

# Makes a copy of a config into a session folder for logging purposes.
def copy_config(config, session, additional_info=None):
    try: os.mkdir(str(session))
    except: pass
    if additional_info is not None:
        config["additional_info"] = additional_info
    save_config(config, os.path.join(str(session), "config.json"))

# Creates a Trainer object from the config.json file.
def make_trainer(config, session):
    loader = lambda : getattr(loaders, config["loader"]["name"])(**config["loader"]["args"])
    return Trainer(session, loader, **config["trainer"]["args"])

# Creates a model from the config.json file.
def make_model(config):
    return getattr(models, config["model"]["name"])(**config["model"]["args"])

# Generates a chain thaw policy given the number of network layers.
def get_chain_thaw_policy(n_layers=8):
    # 1) Freeze every layer except the last (softmax) layer and train it.
    # 2) Freeze every layer except the first layer and train it.
    # 3) Freeze every layer except the second etc., until the second last layer.
    # 4) Unfreeze all layers and train entire model.
    layers = n_layers
    policy = [[False]*(layers-1) + [True]]
    policy += [[False] * (i-1) + [True] + [False] * (layers-i) for i in range(1, layers)]
    policy.append([True]*layers)
    return policy

# Generates a gradual unfreezing policy given the number of network layers.
def get_gradual_unfreezing_policy(n_layers=8):
    # 1) Freeze every layer except the last (softmax) layer and train it.
    # 2) Keeping the last layer unfrozen, also unfreeze the second-to-last layer.
    # 3) Continue until all layers are unfrozen.
    return [[False]*(n_layers-i) + [True]*i for i in range(1,n_layers+1)]