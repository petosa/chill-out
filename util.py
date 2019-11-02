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
        for _, w1 in w.named_children():
            #These loops go over all nested children in the model architecture (including those without grad updates)
            for l, _ in w1.named_parameters():
                #This loop filters out any children that aren't trainable, but we only want to count the layer if theres at least 1 trainable node within it.
                if w1 not in layers:
                    layers.append(w1)
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
            if self.verbose: print(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).')
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

# Creates a Trainer object from the config.json file.
def make_trainer(session):
    config = json.load(open("config.json", "r"))
    loader = lambda : getattr(loaders, config["loader"]["name"])(**config["loader"]["args"])
    return Trainer(session, loader, **config["trainer"]["args"])

# Creates a model from the config.json file.
def make_model():
    config = json.load(open("config.json", "r"))
    return getattr(models, config["model"]["name"])(**config["model"]["args"])

# Makes a copy of a config into a session folder for logging purposes.
def copy_config(session, additional_info=None):
    try: os.mkdir(str(session))
    except: pass
    config = json.load(open("config.json", "r"))
    if additional_info is not None:
        config["additional_info"] = additional_info
    json.dump(config, open(os.path.join(str(session), "config.json"), "w"), indent=4)