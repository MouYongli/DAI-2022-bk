import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict

from .torchbnn.linear import BayesLinear
from .torchbnn.batchnorm import BayesBatchNorm2d, BayesBatchNorm1d
from .torchbnn.conv import BayesConv2d

class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)
        x = self.fc3(x)
        return x


class DigitBayesModel(nn.Module):
    """
    Model for benchmark experiment on Bayes Digits.
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitBayesModel, self).__init__()
        self.conv1 = BayesConv2d(3, 64, 5, 1, 2)
        self.bn1 = BayesBatchNorm2d(64)
        self.conv2 = BayesConv2d(64, 64, 5, 1, 2)
        self.bn2 = BayesBatchNorm2d(64)
        self.conv3 = BayesConv2d(64, 128, 5, 1, 2)
        self.bn3 = BayesBatchNorm2d(128)
        self.fc1 = BayesLinear(6272, 2048)
        self.bn4 = BayesBatchNorm1d(2048)
        self.fc2 = BayesLinear(2048, 512)
        self.bn5 = BayesBatchNorm1d(512)
        self.fc3 = BayesLinear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)
        x = func.relu(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (BayesLinear, BayesConv2d, BayesBatchNorm2d, BayesBatchNorm1d)):
                m.reset_parameters()

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),

                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # x = torch.randn((5,3,28,28))
    # m = DigitBayesModel()
    # y = m(x)

    # model = DigitModel()
    # print("============== state_dict() =================")
    # for key in model.state_dict().keys():
    #     print(key)
    #
    # print("============== state_dict() =================")
    # for name, param in model.named_parameters():
    #     print(name)

    model = DigitBayesModel()
    # print("============== state_dict() =================")
    # for key in model.state_dict().keys():
    #     print(key)
    #
    # print("============== state_dict() =================")
    # for name, param in model.named_parameters():
    #     print(name)

    module_set = set([name.split("_")[0] for name, param in model.named_parameters()])
    print(len(module_set))

    print(len([name.split("_")[0] for name, param in model.named_parameters()]))