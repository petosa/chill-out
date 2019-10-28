
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch_lr_finder import LRFinder

import torchvision
import os
import util
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from util import EarlyStopping


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Trainer:

    def __init__(self, session, data_loader, batch_size=256, val_percentage=.2, data_folder="data", no_cuda=False, seed=0, verbose=False):
        os.mkdir(str(session))
        self.session = str(session)
        self.verbose = verbose
        self.batch_size = batch_size
        self.val_percentage = val_percentage
        self.data_folder = data_folder
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.seed = seed
        self.log_file = str(session) + "/log.txt"
        seed_torch(seed=self.seed)
        self.log_line("num_epoch, train_loss, train_acc, valid loss, valid_test")
        #data_loader()
        self.load_cifar()

    def load_cifar(self):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        datapath = os.path.join(self.data_folder, "cifar10")
        trainset = torchvision.datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
        idx = list(range(len(trainset)))
        np.random.shuffle(idx)
        idx = idx[:1000]
        val_percentage = self.val_percentage
        split_idx = int(val_percentage*len(idx))
        val_idx = idx[-split_idx:]
        train_idx = idx[:-split_idx]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=val_sampler,
            num_workers=4, pin_memory=True
        )
        testset = torchvision.datasets.CIFAR10(datapath, train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )
        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_classes = 10
        self.criterion = F.cross_entropy


    def train(self, model, optimizer, source, destination, policy_step, patience=3):

        self.log_line("Model {}, policy step {}.".format(source, policy_step))
        util.full_load(model, optimizer, source, self.session)

        if self.verbose:
            print ("Model reloaded from: {}".format(os.path.join(self.session, str(source) + ".pt")))
            print ("Current Policy: " + str(policy_step))
        if self.cuda:
            model.cuda()

        for i, l in enumerate(util.get_trainable_layers(model)):
            for _, p in l.named_parameters():
                p.requires_grad = policy_step[i]
        
        early_stopping = EarlyStopping(patience=patience, verbose=True)


        import torch
        optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, nesterov=True, weight_decay=1e-4)
        lr_finder = LRFinder(model, optimizer, self.criterion, device="cuda")
        lr_finder.range_test(self.train_loader, end_lr=1e-1, num_iter=60, smooth_f=0.0, diverge_th=3)
        hist = np.array(lr_finder.history["loss"])
        lrs = np.array(lr_finder.history["lr"])
        best_lr = lrs[np.argmin(hist)]/3

        '''
        import matplotlib.pyplot as plt
        plt.plot(lrs, hist)
        plt.axvline(best_lr)
        plt.xscale("log")
        plt.legend()
        plt.show()
        '''
        
        print("Found best learning rate: ", best_lr)
        self.log_line("Selected learning rate of {}.".format(best_lr))
        lr_finder.reset()

        #optimizer = torch.optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-4)
        optimizer = torch.optim.SGD(model.parameters(), best_lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

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
            train_loss, train_acc = self.evaluate(model, 'train', n_batches=3)
            val_loss, val_acc = self.evaluate(model, 'val')

            self.log_line("{},{:.6f},{},{:.6f},{}".format(epoch, train_loss, train_acc, val_loss, val_acc))
            if self.verbose:
                print('Train Epoch: {} \tTrain Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(epoch, train_loss, val_loss, val_acc))
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping.update(val_loss, model, optimizer)
            
            if early_stopping.early_stop:
                print("Early stopping, epoch:", i)
                break
            
            epoch += 1
        

        model.load_state_dict(early_stopping.best_model)
        optimizer.load_state_dict(early_stopping.best_optim)

        util.full_save(model, optimizer, destination, self.session)

        if self.verbose:
            print("Saved model: {}".format(os.path.join(self.session, str(destination) + ".pt")))

        self.log_line("Model saved at {}".format(os.path.join(self.session, str(destination) + ".pt")))
        self.log_line("Final Valid Loss: {:.6f}, Final Valid Acc: {}".format(val_loss, val_acc))

        return val_loss


    def evaluate(self, model, split, n_batches=None):
        model.eval()
        loss = 0
        correct = 0
        n_examples = 0
        if split == "val":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        elif split == "train":
            loader = self.train_loader
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
        loss /= n_examples
        acc = 100. * correct / n_examples
        return loss, acc

    
    def log_line(self, line):
        with open(self.log_file, "a+") as f:
            f.write(str(line) + "\n")

