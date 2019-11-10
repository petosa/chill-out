import torchvision
import torch
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



def load_cifar(batch_size=256, network_train_size=.2, search_train_size=.2, data_folder="data"):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    datapath = os.path.join(data_folder, "cifar10")
    trainset = torchvision.datasets.CIFAR10(datapath, train=True, download=True, transform=transform)
    idx = list(range(len(trainset)))
    np.random.shuffle(idx)

    nt_size = int(network_train_size*len(idx)) if network_train_size < 1 else network_train_size
    st_size = int(search_train_size*len(idx)) if search_train_size < 1 else search_train_size
    val_size = len(idx) - nt_size - st_size
    print("Network train size:", nt_size, "Search train size:", st_size, "Val size:", val_size)
    print("Total size:", len(idx))

    nt_idx = idx[:nt_size]
    st_idx = idx[nt_size:nt_size+st_size]
    val_idx = idx[nt_size+nt_size:]
    print("Network train size:", len(nt_idx), "Search train size:", len(st_idx), "Val size:", len(val_idx))

    network_train_sampler = SubsetRandomSampler(nt_idx)
    search_train_sampler = SubsetRandomSampler(st_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    network_train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=network_train_sampler,
        num_workers=4, pin_memory=True
    )
    search_train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=search_train_sampler,
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
    return network_train_loader, search_train_loader, val_loader, test_loader, criterion