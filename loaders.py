import torchvision
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



def load_cifar(batch_size=256, val_size=.2, size_limit=None, data_folder="data"):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    datapath = os.path.join(data_folder, "cifar10")
    trainset = torchvision.datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
    idx = list(range(len(trainset)))
    np.random.shuffle(idx)
    idx = idx if size_limit is None else idx[:size_limit]
    split_idx = int(val_size*len(idx)) if val_size < 1 else val_size
    val_idx = idx[-split_idx:]
    train_idx = idx[:-split_idx]
    print(len(val_idx), len(train_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True
    )
    testset = torchvision.datasets.CIFAR10(datapath, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    criterion = F.cross_entropy
    return train_loader, val_loader, test_loader, criterion