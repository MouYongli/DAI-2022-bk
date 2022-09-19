import os
import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from synthetic_digits import SyntheticDigits
from mnistm import MNISTM

here = osp.dirname(osp.abspath(__file__))
data_dir = osp.join(here, "../../Data")
if not osp.exists(data_dir):
    os.makedirs(data_dir)

def imshow(img_list, lbl_list, ds_list):
    for i in range(len(img_list)):
        img = img_list[i] / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.subplot(len(img_list), 1, i+1)
        plt.title("{}: {}".format(ds_list[i], lbl_list[i].numpy()))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis("off")
        plt.tight_layout()
    plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 8

# cifar10_trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True,  download=True, transform=transform)
# cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=batch_size, shuffle=True)
# cifar10_testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
# cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=batch_size, shuffle=False)

trainset_mnist = torchvision.datasets.MNIST(root=osp.join(data_dir, 'MNIST'), train=True,  download=True, transform=transform)
trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist, batch_size=batch_size, shuffle=True)
testset_mnist = torchvision.datasets.MNIST(root=osp.join(data_dir, 'MNIST'), train=False,  download=True, transform=transform)
testloader_mnist = torch.utils.data.DataLoader(testset_mnist, batch_size=batch_size, shuffle=False)

trainset_svhn = torchvision.datasets.SVHN(root=osp.join(data_dir, 'SVHN'), split="train",  download=True, transform=transform)
trainloader_svhn = torch.utils.data.DataLoader(trainset_svhn, batch_size=batch_size, shuffle=True)
testset_svhn = torchvision.datasets.SVHN(root=osp.join(data_dir, 'SVHN'), split="test",  download=True, transform=transform)
testloader_svhn = torch.utils.data.DataLoader(testset_svhn, batch_size=batch_size, shuffle=False)

trainset_usps = torchvision.datasets.USPS(root=osp.join(data_dir, 'USPS'), train=True,  download=True, transform=transform)
trainloader_usps = torch.utils.data.DataLoader(trainset_usps, batch_size=batch_size, shuffle=True)
testset_usps = torchvision.datasets.USPS(root=osp.join(data_dir, 'USPS'), train=False,  download=True, transform=transform)
testloader_usps = torch.utils.data.DataLoader(testset_usps, batch_size=batch_size, shuffle=False)

trainset_synthetic_digits = SyntheticDigits(root=osp.join(data_dir, 'SyntheticDigits'), train=True, download=True, transform=transform)
trainloader_synthetic_digits = torch.utils.data.DataLoader(trainset_synthetic_digits, batch_size=batch_size, shuffle=True)
testset_synthetic_digits = SyntheticDigits(root=osp.join(data_dir, 'SyntheticDigits'), train=False, download=True, transform=transform)
testloader_synthetic_digits = torch.utils.data.DataLoader(testset_synthetic_digits, batch_size=batch_size, shuffle=False)

trainset_mnistm = MNISTM(root=osp.join(data_dir, 'MNISTM'), train=True,  download=True, transform=transform)
trainloader_mnistm = torch.utils.data.DataLoader(trainset_mnistm, batch_size=batch_size, shuffle=True)
testset_mnistm = MNISTM(root=osp.join(data_dir, 'MNISTM'), train=False,  download=True, transform=transform)
testloader_mnistm = torch.utils.data.DataLoader(testset_mnistm, batch_size=batch_size, shuffle=False)

# get some random training images
dataiter_mnist = iter(trainloader_mnist)
images_mnist, labels_mnist = dataiter_mnist.next()
dataiter_svhn = iter(trainloader_svhn)
images_svhn, labels_svhn = dataiter_svhn.next()
dataiter_usps = iter(trainloader_usps)
images_usps, labels_usps = dataiter_usps.next()
dataiter_synthetic_digits = iter(trainloader_synthetic_digits)
images_synthetic_digits, labels_synthetic_digits = dataiter_synthetic_digits.next()
dataiter_mnistm = iter(trainloader_mnistm)
images_mnistm, labels_mnistm = dataiter_mnistm.next()

# imshow([torchvision.utils.make_grid(images_mnist), torchvision.utils.make_grid(images_svhn),
#         torchvision.utils.make_grid(images_usps), torchvision.utils.make_grid(images_synthetic_digits),
#         torchvision.utils.make_grid(images_mnistm)],
#        [labels_mnist, labels_svhn, labels_usps, labels_synthetic_digits, labels_mnistm],
#        ["MNIST", "SVHN", "USPS", "Synthetic Digits", "MNISTM"])

print("Number of Data in each Training Set:")
print("MNIST", len(trainset_mnist))
print("SVHN", len(trainset_svhn))
print("USPS", len(trainset_usps))
print("Synthetic Digits", len(trainset_synthetic_digits))
print("MNIST-M", len(trainset_mnistm))

print("Number of Data in each Test Set:")
print("MNIST", len(testset_mnist))
print("SVHN", len(testset_svhn))
print("USPS", len(testset_usps))
print("Synthetic Digits", len(testset_synthetic_digits))
print("MNIST-M", len(testset_mnistm))

"""
# Number of Data in each Training Set:
# MNIST 60000
# SVHN 73257
# USPS 7291
# Synthetic Digits 479400
# MNIST-M 60000
# Number of Data in each Test Set:
# MNIST 10000
# SVHN 26032
# USPS 2007
# Synthetic Digits 9553
# MNIST-M 10000
"""