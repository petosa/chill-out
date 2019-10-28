import torchvision.models as models
from torch.utils.model_zoo import load_url
from torch import nn, save, load
import numpy as np
import os
from copy import deepcopy


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


def load_alexnet(num_classes):
    url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    model = models.AlexNet()
    state_dict = load_url(url, progress=True)
    model.load_state_dict(state_dict)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features,num_classes)
    return model

def load_squeezenet(num_classes):
    url = "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth"
    model = models.SqueezeNet(version=1.1)
    state_dict = load_url(url, progress=True)
    model.load_state_dict(state_dict)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    return model

def full_save(model, optimizer, id, session):
    try: os.mkdir(str(session))
    except: pass
    data = {"model":model.state_dict(), "optim":optimizer.state_dict()}
    save(data, os.path.join(str(session), str(id) + ".pt"))

def full_load(model, optimizer, id, session):
    state_dict = load(os.path.join(str(session), str(id) + ".pt"))
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optim"])


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.best_model = None
        self.best_optim = None
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = np.inf
        self.early_stop = False
        self.delta = delta

    def update(self, val_loss, model, optimizer):
        if self.best_val_loss == np.inf or self.best_val_loss > val_loss:
            if self.verbose: print(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).')
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        self.best_model = deepcopy(model.state_dict())
        self.best_optim = deepcopy(optimizer.state_dict())