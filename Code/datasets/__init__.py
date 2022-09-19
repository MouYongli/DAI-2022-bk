import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .synthetic_digits import SyntheticDigits
from .mnistm import MNISTM

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, "../../Data")

num_total_mnist = 60000
num_total_svhn = 73257
num_total_usps = 7291
num_total_synthetic_digits = 479400
num_total_mnistm = 60000

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def get_dataloaders(args):
    """
    :param args:
    :return: training dataloaders and test dataloaders
    """
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    idxs_mnist = np.random.choice(num_total_mnist, args.num_data_mnist, replace=False)
    idxs_svhn = np.random.choice(num_total_svhn, args.num_data_svhn, replace=False)
    idxs_usps = np.random.choice(num_total_usps, args.num_data_usps, replace=False)
    idxs_synthetic_digits = np.random.choice(num_total_synthetic_digits, args.num_data_synth, replace=False)
    idxs_mnistm = np.random.choice(num_total_mnistm, args.num_data_mnistm, replace=False)

    trainset_mnist = DatasetSplit(torchvision.datasets.MNIST(root=osp.join(data_dir, 'MNIST'), train=True, download=True,
                                                transform=transform_mnist), idxs_mnist)
    trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=args.batch_size, shuffle=True)
    testset_mnist = torchvision.datasets.MNIST(root=osp.join(data_dir, 'MNIST'), train=False, download=True,
                                               transform=transform_mnist)
    testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=args.batch_size, shuffle=False)

    trainset_svhn = DatasetSplit(torchvision.datasets.SVHN(root=osp.join(data_dir, 'SVHN'), split="train", download=True,
                                              transform=transform_svhn), idxs_svhn)
    trainloader_svhn = torch.utils.data.DataLoader(trainset_svhn, batch_size=args.batch_size, shuffle=True)
    testset_svhn = torchvision.datasets.SVHN(root=osp.join(data_dir, 'SVHN'), split="test", download=True,
                                             transform=transform_svhn)
    testloader_svhn = torch.utils.data.DataLoader(testset_svhn, batch_size=args.batch_size, shuffle=False)

    trainset_usps = DatasetSplit(torchvision.datasets.USPS(root=osp.join(data_dir, 'USPS'), train=True, download=True,
                                              transform=transform_usps), idxs_usps)
    trainloader_usps = torch.utils.data.DataLoader(trainset_usps, batch_size=args.batch_size, shuffle=True)
    testset_usps = torchvision.datasets.USPS(root=osp.join(data_dir, 'USPS'), train=False, download=True,
                                             transform=transform_usps)
    testloader_usps = torch.utils.data.DataLoader(testset_usps, batch_size=args.batch_size, shuffle=False)

    trainset_synthetic_digits = DatasetSplit(SyntheticDigits(root=osp.join(data_dir, 'SyntheticDigits'), train=True, download=True,
                                                transform=transform_synth), idxs_synthetic_digits)
    trainloader_synthetic_digits = torch.utils.data.DataLoader(trainset_synthetic_digits, batch_size=args.batch_size,
                                                               shuffle=True)
    testset_synthetic_digits = SyntheticDigits(root=osp.join(data_dir, 'SyntheticDigits'), train=False, download=True,
                                               transform=transform_synth)
    testloader_synthetic_digits = torch.utils.data.DataLoader(testset_synthetic_digits, batch_size=args.batch_size,
                                                              shuffle=False)

    trainset_mnistm = DatasetSplit(MNISTM(root=osp.join(data_dir, 'MNISTM'), train=True, download=True, transform=transform_mnistm), idxs_mnistm)
    trainloader_mnistm = torch.utils.data.DataLoader(trainset_mnistm, batch_size=args.batch_size, shuffle=True)
    testset_mnistm = MNISTM(root=osp.join(data_dir, 'MNISTM'), train=False, download=True, transform=transform_mnistm)
    testloader_mnistm = torch.utils.data.DataLoader(testset_mnistm, batch_size=args.batch_size, shuffle=False)

    train_loaders = [trainloader_mnist, trainloader_svhn, trainloader_usps, trainloader_synthetic_digits, trainloader_mnistm]
    test_loaders  = [testloader_mnist, testloader_svhn, testloader_usps, testloader_synthetic_digits, testloader_mnistm]

    return train_loaders, test_loaders