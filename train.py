import time
import alexnet
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from policies.gradual_unfreezing import get_gradual_unfreezing_policy
from policies.chain_thaw import get_chain_thaw_policy

class PolicyEvaluator:
    def __init__(self, model_class=alexnet.AlexNet, lr=1e-4, momentum=.9, weight_decay=0, batch_size=256,
                    epochs=10, epochs_between_states=10, 
                    val_percentage=.2, no_cuda=False, seed=1, 
                    log_interval=10, cifar10_dir="data", log_file="log.txt", verbose=False):

        self.log_file = log_file
        self.model_class = model_class
        self.verbose = verbose
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.epochs_between_states = epochs_between_states
        self.val_percentage = val_percentage
        self.cuda = not no_cuda and torch.cuda.is_available()
        self.seed = seed
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        
        self.log_interval = log_interval
        self.cifar10_dir = cifar10_dir
        self.load_cifar()

        with open(self.log_file, "a+") as fh:
            fh.write('num_epoch, train_loss, train_acc, valid loss, valid_test\n')
    
    def load_cifar(self):
        #Load data
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        trainset = torchvision.datasets.CIFAR10(self.cifar10_dir, train=True,
                                                download=True, transform=transform)
        idx = list(range(len(trainset)))
        np.random.shuffle(idx)
        val_percentage = self.val_percentage
        split_idx = int(val_percentage*len(trainset))
        val_idx = idx[-split_idx:]
        train_idx = idx[:-split_idx]
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=train_sampler,
            num_workers=2, pin_memory=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, sampler=val_sampler,
            num_workers=2, pin_memory=True,
        )

        testset = torchvision.datasets.CIFAR10(self.cifar10_dir, train=False,
                                            download=True, transform=transform)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_classes = 10
        self.criterion = F.cross_entropy


    def train(self, model_path, destination_model_path, policy_step):
        with open(self.log_file, "a+") as fh:
            fh.write(model_path + '\n')
            fh.write(str(policy_step) + '\n')

        model = self.model_class(num_classes=self.n_classes)
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

        if self.verbose:
            print ('Model reloaded from: {}'.format(model_path))
            print(model)
            print ("Current Policy : " + str(policy_step))
        if self.cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # model.named_parameters = [layer1.weight, layer1.bias, layer2.weight....]
        for i, (layer, w) in enumerate(model.named_parameters()):
            w.requires_grad = policy_step[i // 2] # weight and bias are in named_parameters, not in policy
        for i in range(self.epochs):
            model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                images, targets = Variable(batch[0]), Variable(batch[1])
                if self.cuda:
                    images, targets = images.cuda(), targets.cuda()
                model.zero_grad()
                output = model(images)
                loss = self.criterion(output, targets)
                loss.backward()
                optimizer.step()

            train_loss, train_acc = self.evaluate(model, 'train')
            val_loss, val_acc = self.evaluate(model, 'val')
            with open(self.log_file, "a+") as fh:
                fh.write('{},{:.6f},{},{:.6f},{}\n'.format(i+1, train_loss, train_acc, val_loss, val_acc))
            if self.verbose:
                print('Train Epoch: {} \t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                i + 1, train_loss, val_loss, val_acc))
        test_loss, test_acc = self.evaluate(model, 'test', verbose=True)

        
        torch.save(model.state_dict(), destination_model_path)
        if self.verbose:
            print ('New model saved at: {}'.format(destination_model_path))
        with open(self.log_file, "a+") as fh:
            fh.write('Model saved at {}\n'.format(destination_model_path))
            fh.write('Test Loss: {:.6f}, Test Acc: {}\n'.format(test_loss, test_acc))

        return test_loss


    def evaluate(self, model, split, verbose=False, n_batches=None):
        model.eval()
        loss = 0
        correct = 0
        n_examples = 0
        if split == 'val':
            loader = self.val_loader
        elif split == 'test':
            loader = self.test_loader
        elif split == 'train':
            loader = self.train_loader
        for batch_i, batch in enumerate(loader):
            data, target = batch
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = model(data)
            loss += self.criterion(output, target, size_average=False).data
            # predict the argmax of the log-probabilities
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            n_examples += pred.size(0)
            if n_batches and (batch_i >= n_batches):
                break

        loss /= n_examples
        acc = 100. * correct / n_examples
        if verbose:
            print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                split, loss, correct, n_examples, acc))
        return loss, acc


    




'''
parser = argparse.ArgumentParser(description='Chill-out')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=1e-4,
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M', default=.9,
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=256,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=10,
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['alexnet'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
parser.add_argument('--epochs-between-states', type=int, help='idfk')
parser.add_argument('--val-percentage', type=float, default=.2, help='idfk')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--cifar10-dir', default='data',
                    help='directory that contains cifar-10-batches-py/ '
                         '(downloaded automatically if necessary)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
'''


