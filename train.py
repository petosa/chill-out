import os
import util
import torch
import random
import numpy as np
import torch.optim as optim
from torch_lr_finder.lr_finder import LRFinder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader 
import util


class Trainer:

    def __init__(self, session, loader, fixed_lr=None, halve=False, no_cuda=False, seed=None, verbose=True):
        
        # Seed execution
        if seed is not None:
            util.seed_torch(seed=seed)

        # Class variables
        self.session = str(session)
        self.verbose = verbose
        self.fixed_lr = fixed_lr
        self.halve = halve
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.seed = seed
        self.log_file = str(session) + "/log.txt"
        self.train_loader, self.val_loader, self.test_loader, self.criterion = loader()

        # Halve
        self.val2_loader = None
        if self.halve:
            val_idx = self.val_loader.sampler.indices
            np.random.shuffle(val_idx)
            half = int(len(val_idx)/2)
            val1_idx, val2_idx = val_idx[:half], val_idx[half:]
            print("val1_idx", len(val1_idx), "val2_idx", len(val2_idx))
            val1_sampler, val2_sampler = SubsetRandomSampler(val1_idx), SubsetRandomSampler(val2_idx)
            dataset, batch_size = self.val_loader.dataset, self.val_loader.batch_size
            self.val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val1_sampler, num_workers=4, pin_memory=True)
            self.val2_loader = DataLoader(dataset, batch_size=batch_size, sampler=val2_sampler, num_workers=4, pin_memory=True)


    # Train a model
    def train(self, model, source, destination, freeze_state, patience=3, weight_decay=1e-4):

        # Restore model from checkpoint
        util.full_load(model, source, self.session)
        self.log_line("Base model {}, policy step {}.".format(source, freeze_state))
        if self.verbose:
            print("Model reloaded from: {}".format(os.path.join(self.session, str(source) + ".pt")))
            print("Current Policy: " + str(freeze_state))

        # Freeze the specified layers
        for i, l in enumerate(util.get_trainable_layers(model)):
            for _, p in l.named_parameters():
                p.requires_grad = freeze_state[i]
        
        # Initialize early stopper
        early_stopping = util.EarlyStopping(patience=patience, verbose=self.verbose)

        # Find and set best learning rate
        if self.fixed_lr is None:
            optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, nesterov=True, weight_decay=weight_decay)
            lr_finder = LRFinder(model, optimizer, self.criterion, device="cuda" if self.cuda else "cpu", verbose=False)
            lr_finder.range_test(self.train_loader, end_lr=1e-1, num_iter=60, smooth_f=0.0, diverge_th=3)
            hist = np.array(lr_finder.history["loss"])
            lrs = np.array(lr_finder.history["lr"])
            best_lr = lrs[np.argmin(hist)]/3
            lr_finder.reset()
        else:
            best_lr = self.fixed_lr

        if self.verbose: print("Found best learning rate: ", best_lr)
        self.log_line("Selected learning rate of {}.".format(best_lr))
        optimizer = torch.optim.SGD(model.parameters(), best_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)

        # Train to convergence
        epoch = 1
        while True:
            model.train()
            for images, targets in self.train_loader:
                if self.cuda:
                    images, targets = images.cuda(), targets.cuda()
                model.zero_grad()
                output = model(images)
                loss = self.criterion(output, targets)
                loss.backward()
                optimizer.step()
            train_loss, train_acc = self.evaluate(model, self.train_loader, n_batches=3)
            val_loss, val_acc = self.evaluate(model, self.val_loader)
            self.log_line("{},{:.6f},{},{:.6f},{}".format(epoch, train_loss, train_acc, val_loss, val_acc))
            if self.verbose:
                print('Train Epoch: {} Train Loss: {:.6f} Val Loss: {:.6f} Val Acc: {}'.format(epoch, train_loss, val_loss, val_acc))
            early_stopping.update(val_loss, model)
            if early_stopping.early_stop:
                if self.verbose: print("Early stopping, epoch:", epoch)
                break
            epoch += 1
        
        # Restore and save best model
        model.load_state_dict(early_stopping.best_model)
        util.full_save(model, destination, self.session)
        train_loss, train_acc = self.evaluate(model, self.train_loader)
        val_loss, val_acc = self.evaluate(model, self.val_loader)
        if self.verbose: print("Saved model: {}".format(os.path.join(self.session, str(destination) + ".pt")))
        self.log_line("Model saved at {}".format(os.path.join(self.session, str(destination) + ".pt")))
        self.log_line("Final Train Loss: {:.6f}, Final Train Acc: {}".format(train_loss, train_acc))
        self.log_line("Final Val Loss: {:.6f}, Final Val Acc: {}".format(val_loss, val_acc))
        if self.halve:
            val2_loss, val2_acc = self.evaluate(model, self.val2_loader)
            self.log_line("Final Val 2 Loss: {:.6f}, Final Val 2 Acc: {}".format(val2_loss, val2_acc))
        
        torch.cuda.empty_cache()
        return val_loss


    # Evaluates loss, accuracy for one of the data splits.
    def evaluate(self, model, loader, n_batches=None):
        model.eval()
        loss, correct, n_examples = 0, 0, 0
        for batch_i, (images, targets) in enumerate(loader):
            if self.cuda:
                images, targets = images.cuda(), targets.cuda()
            with torch.no_grad():
                output = model(images)
                loss += self.criterion(output, targets, reduction="sum").data
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum()
                n_examples += pred.size(0)
                if n_batches and (batch_i >= n_batches):
                    break
        if n_examples == 0: return 0., 0.
        loss /= n_examples
        acc = 100. * correct / n_examples
        return loss, acc

    
    # Writes the given line to the log file
    def log_line(self, line):
        with open(self.log_file, "a+") as f:
            f.write(str(line) + "\n")

