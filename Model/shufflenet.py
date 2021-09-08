import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torchsummary
from thop import profile
import os

cfg = {
    'ShuffleNet' : [1, 2, 3, 4, 5],
}
alias = {
    '1' : [144, 288, 576],
    '2' : [200, 400, 800],
    '3' : [240, 480, 960],
    '4' : [272, 544, 1088],
    '5' : [384, 768, 1536],
}

class ShuffleBlock(nn.Module):
    def __init__(self, groups) -> None:
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, channels, height, width = x.size()
        channels_per_group = channels // self.groups

        x = x.view(batch, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, height, width)

        return x

class Stage(nn.Module):
    def __init__(self, in_planes, out_planes, stride, group, shuffle) -> None:
        super(Stage, self).__init__()

        self.stride = stride
        self.shuffle = shuffle

        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else group

        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=g, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=stride, padding=1))

    def forward(self, x):
        print('A : ', x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        print('B : ', out.size())
        if self.shuffle:
            out = self.shuffle1(out)
        print('C : ', out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        print('D : ', out.size())
        out = self.bn3(self.conv3(out))
        print('E : ', out.size())
        res = self.shortcut(x)
        print('F : ', res.size())
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        print('G : ', out.size())

        return out


class ShuffleNet(nn.Module):
    def __init__(self, name, group, shuffle=True, scale=1) -> None:
        super(ShuffleNet, self).__init__()
        self.name = name
        self.group = group
        self.scale = scale

        self.in_planes = 24
        self.num_classes = 10

        self.out_plane = alias[str(cfg[name][int(group) - 1])]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # When using ImageNet dataset
        # self.maxpool = nn.MaxPool2d(3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.stage2 = self._make_layer(4, self.out_plane[0], self.group, shuffle)
        self.stage3 = self._make_layer(8, self.out_plane[1], self.group, shuffle)
        self.stage4 = self._make_layer(4, self.out_plane[2], self.group, shuffle)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(self.out_plane[2], self.num_classes)

    def _make_layer(self, feature_n, out_planes, group, shuffle):
        layers = []
        for i in range(feature_n):
            stride = 2 if i == 0 else 1

            layers.append(Stage(self.in_planes, out_planes, stride=stride, group=group, shuffle=shuffle))
            
        self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        print('1 : ', x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        print('2 : ', out.size())
        out = self.stage2(out)
        print('3 : ', out.size())
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ShuffleNet('ShuffleNet', group=2, shuffle=False)
    model = model.to(device)

    # print(model)

    inp = torch.randn(1, 3, 32, 32).to(device)
 
    torchsummary.summary(model, input_size=(3, 32 ,32))

    macs , params = profile(model, inputs=(inp,))
    print('MACs : ', macs, 'Params : ', params)

if __name__ == '__main__':
    test()