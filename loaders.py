import torchvision
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



def load_cifar(train_size, val_size, batch_size=256, data_folder="data", verbose=True):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    datapath = os.path.join(data_folder, "cifar10")
    trainset = torchvision.datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
    idx = list(range(len(trainset)))
    np.random.shuffle(idx)

    train_size = int(train_size*len(idx)) if train_size < 1 else train_size
    val_size = int(val_size*len(idx)) if val_size < 1 else val_size
    
    assert(train_size+val_size <= len(idx))
    if verbose:
        print("Train size:", train_size, "Val size:", val_size)

    train_idx = idx[:train_size]
    val_idx = idx[-val_size:]
    if verbose:
        print("Train size:", len(train_idx), "Val size:", len(val_idx))

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


def load_scenes(train_size, val_size, batch_size=256, data_folder="data", verbose=True):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    datapath = os.path.join(data_folder, "scene15")
    dataset = torchvision.datasets.ImageFolder(root=datapath, transform=transform)
    idx = list(range(len(dataset)))
    np.random.shuffle(idx)

    train_size = int(train_size*len(idx)) if train_size < 1 else train_size
    val_size = int(val_size*len(idx)) if val_size < 1 else val_size
    test_size = len(dataset)-train_size-val_size
    
    assert(train_size+val_size <= len(idx))
    assert(test_size > 0)
    if verbose:
        print("Train size:", train_size, "Val size:", val_size, "Test size:", test_size)

    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size+val_size]
    test_idx = idx[-test_size]
    if verbose:
        print("Train size:", len(train_idx), "Val size:", len(val_idx), "Test size:", len(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=test_sampler,
        num_workers=4, pin_memory=True
    )
    criterion = F.cross_entropy
    return train_loader, val_loader, test_loader, criterion