import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

dataset_dir = '/home/chanjong/.keras/datasets/'

def train_val_dataset(batch_size, num_workers, valid_size=0.1, shuffle=True):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, download=False, transform=transform_train)

    validset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, download=False, transform=transform_val)

    n_train = len(trainset)
    indices = list(range(n_train))
    split = int(np.floor(valid_size * n_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

 

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    print("Prepare training dataset Complete.. Total size : ", n_train - split)

    validloader = torch.utils.data.DataLoader(
        dataset=validset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    print("Prepare validation dataset Complete.. Total size : ", split)

    return trainloader, validloader


def test_dataset(batch_size, num_workers):

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Prepare test dataset Complete..")

    return testloader

def test():
    trainloader, validloader = train_val_dataset(64)
    testloader = test_dataset(64)

    print(trainloader)

# test()