import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from thop import profile

cfg = {
    'ResNeXt50': ['Bottleneck', [3, 4, 6, 3]],
    'ResNeXt101': ['Bottleneck', [3, 4, 23, 3]],
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, name):
        super(ResNeXt, self).__init__()
        self.in_planes = 64
        self.num_classes = 10

        self.block_name = eval(cfg[name][0])
        self.block_feature = cfg[name][1]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block_name, self.block_feature[0], 64, stride=1)
        self.layer2 = self._make_layer(self.block_name, self.block_feature[1], 128, stride=2)
        self.layer3 = self._make_layer(self.block_name, self.block_feature[2], 256, stride=2)
        self.layer4 = self._make_layer(self.block_name, self.block_feature[3], 512, stride=2)
        self.linear = nn.Linear(512 * self.block_name.expansion, self.num_classes)

    def _make_layer(self, block_name, feature, planes, stride):
        strides = [stride] + [1]*(feature-1)
        layers = []
        for stride in strides:
            layers.append(block_name(self.in_planes, planes, stride))
            self.in_planes = planes * block_name.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out