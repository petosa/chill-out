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

#Load data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.CIFAR10(args.cifar10_dir, train=True,
                                        download=True, transform=transform)

idx = list(range(len(trainset)))
np.random.shuffle(idx)
val_percentage = args.val_percentage
split_idx = int(val_percentage*len(trainset))
val_idx = idx[-split_idx:]
train_idx = idx[:-split_idx]

train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, sampler=train_sampler,
    num_workers=2, pin_memory=True,
)
val_loader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, sampler=val_sampler,
    num_workers=2, pin_memory=True,
)

testset = torchvision.datasets.CIFAR10(args.cifar10_dir, train=False,
                                       download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=True,
    num_workers=2, pin_memory=True,
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_classes = 10


model = alexnet.alexnet(pretrained=True)

# gradual unfreezing
model.train()
criterion = F.cross_entropy
if args.cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
policy = [[False]*(8-i) + [True]*i for i in range(1,9)]
# policy = [
#     [False]*7 + [True],
#     [False]*6 + [True]*2,
#     [False]*5 + [True]*3,
#     [False]*4 + [True]*4,
#     [False]*3 + [True]*5,
#     [False]*2 + [True]*6,
#     [False]*1 + [True]*7,
#     [True]*8,
# ]
print (model)
for step in policy:
    # model.named_parameters = [layer1.weight, layer1.bias, layer2.weight....]
    for i, (layer, w) in enumerate(model.named_parameters()):
        w.requires_grad = step[i // 2] # weight and bias are in named_parameters, not in policy
    print ("Current Policy : " + str(step))
    for i in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            images, targets = Variable(batch[0]), Variable(batch[1])
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()
            model.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        print ("Epoch : " + str(i))
        
torch.save(model.state_dict(), 'model.pt')
