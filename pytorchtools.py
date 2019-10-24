import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = np.inf
        self.early_stop = False
        self.delta = delta

    def update(self, val_loss, model, optimizer):
        if self.best_val_loss == np.inf or self.best_val_loss > val_loss:
            if self.verbose: print(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}).')
            self.save_checkpoint(val_loss, model, optimizer)
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'checkpoint.pt')
        torch.save(optimizer.state_dict(), 'optimizer.pt')
