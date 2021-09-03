import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from thop import profile
import os

cfg = {
    'ResNeXt29': [3, 3, 3],
    'ResNeXt50': [3, 4, 6, 3],
    'ResNeXt101': [3, 4, 23, 3],
}

class Block(nn.Module):
    expansion = 2

    def __init__(self, in_planes, cardinality=32, width=4, stride=1):
        super(Block, self).__init__()

        group_width = cardinality * width

        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, name, cardinality, width):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = width
        self.in_planes = 64
        self.num_classes = 10

        self.block_feature = cfg[name]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block_feature[0], stride=1)
        self.layer2 = self._make_layer(self.block_feature[1], stride=2)
        self.layer3 = self._make_layer(self.block_feature[2], stride=2)
        # for ResNeXt50, 101
        if len(self.block_feature) == 4:
            self.layer4 = self._make_layer(self.block_feature[3], stride=2)
        # use Adaptive Average Pooling instead of Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 8))
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width * 8, self.num_classes)

    def _make_layer(self, feature, stride):
        strides = [stride] + [1]*(feature-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if len(self.block_feature) == 4:
            out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNeXt('ResNeXt101', cardinality=32, width=4)
    model = model.to(device)

    inp = torch.randn(1, 3, 32, 32).to(device)
 
    torchsummary.summary(model, input_size=(3, 32 ,32))

    macs , params = profile(model, inputs=(inp,))
    print('MACs : ', macs, 'Params : ', params)

if __name__ == '__main__':
    test()