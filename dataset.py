import torch
import torchvision
import torchvision.transforms as transforms

dataset_dir = '/home/chanjong/.keras/datasets/'

def training_dataset(batch_size, num_workers):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=dataset_dir, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Prepare training dataset Complete..")

    return trainloader

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
    trainloader = training_dataset(64)
    testloader = test_dataset(64)

    print(trainloader)

# test()