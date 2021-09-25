import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
import torchsummary
from thop import profile
import os

cfg = {
    'ShuffleNet' : [1, 2, 3, 4, 8],
    'ShuffleNetV2' : [0.5, 1, 1.5, 2]
}
alias = {
    '1' : [144, 288, 576],
    '2' : [200, 400, 800],
    '3' : [240, 480, 960],
    '4' : [272, 544, 1088],
    '8' : [384, 768, 1536],
}

aliasV2 = {
    '0.5' : [48, 96, 192, 1024],
    '1'   : [116, 232, 464, 1024],
    '1.5' : [176, 352, 704, 1024],
    '2'   : [244, 488, 976, 2048],
}

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2) -> None:
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, n_channels, height, width = x.size()
        channels_per_group = n_channels // self.groups

        x = x.reshape(batch, self.groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch, n_channels, height, width)

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
        out = F.relu(self.bn1(self.conv1(x)))
        if self.shuffle:
            out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)

        return out

class StageV2(nn.Module):
    def __init__(self, in_planes, out_planes, stride, shuffle) -> None:
        super(StageV2, self).__init__()

        self.stride = stride
        self.shuffle = shuffle

        mid_planes = out_planes // 2

        # left layers
        self.convL1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bnL1 = nn.BatchNorm2d(in_planes)
        self.convL2 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bnL2 = nn.BatchNorm2d(mid_planes)

        # right layers
        self.convR1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bnR1 = nn.BatchNorm2d(mid_planes)
        self.convR2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bnR2 = nn.BatchNorm2d(mid_planes)
        self.convR3 = nn.Conv2d(mid_planes, mid_planes, kernel_size=1, bias=False)
        self.bnR3 = nn.BatchNorm2d(mid_planes)

        self.shuffle1 = ShuffleBlock(groups=mid_planes)

    def forward(self, x):
        left = self.bnL1(self.convL1(x))
        left = F.relu(self.bnL2(self.convL2(left)))

        right = F.relu(self.bnR1(self.convR1(x)))
        right = self.bnR2(self.convR2(right))
        right = F.relu(self.bnR3(self.convR3(right)))

        out = torch.cat([left, right], 1) if self.stride == 2 else left + right

        print(out.size())

        if self.shuffle:
            out = self.shuffle1(out)

        return out

class ShuffleNet(nn.Module):
    def __init__(self, name, group, shuffle=True, scale=1) -> None:
        super(ShuffleNet, self).__init__()
        self.name = name
        self.group = group
        self.scale = scale

        self.in_planes = 24
        self.num_classes = 10
        if group in cfg[name]:
            self.out_plane = alias[str(group)]
        else:
            raise ValueError("Group {} is not Supported".format(group))

        self.feature_n = [4, 8, 4]
        if scale != 1:
            self.feature_n = [int(i * scale) for i in self.feature_n]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # When using ImageNet dataset
        # self.maxpool = nn.MaxPool2d(3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.stage2 = self._make_layer(self.feature_n[0], self.out_plane[0], group=group, shuffle=shuffle)
        self.stage3 = self._make_layer(self.feature_n[1], self.out_plane[1], group=group, shuffle=shuffle)
        self.stage4 = self._make_layer(self.feature_n[2], self.out_plane[2], group=group, shuffle=shuffle)
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(self.out_plane[2], self.num_classes)

    def _make_layer(self, feature_n, out_planes, group, shuffle):
        layers = []
        for i in range(feature_n):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0

            layers.append(Stage(self.in_planes, out_planes - cat_planes, stride=stride, group=group, shuffle=shuffle))
            
            self.in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, name, c_size, shuffle=True) -> None:
        super(ShuffleNetV2, self).__init__()
        self.name = name

        self.in_planes = 24
        self.num_classes = 10
        if c_size in cfg[name]:
            self.out_plane = aliasV2[str(c_size)]
        else:
            raise ValueError("Group {} is not Supported".format(c_size))
        
        # stride 2 -> 1 for cifar dataset
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        
        # When using ImageNet dataset
        # self.maxpool = nn.MaxPool2d(3, stride=2)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.stage2 = self._make_layer(3, self.out_plane[0], shuffle)
        self.stage3 = self._make_layer(7, self.out_plane[1], shuffle)
        self.stage4 = self._make_layer(3, self.out_plane[2], shuffle)
        self.conv5 = nn.Conv2d(self.out_plane[2], self.out_plane[3], kernel_size=1, stride=1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.out_plane[3])
        self.avg_pool = nn.AvgPool2d(4)
        self.linear = nn.Linear(self.out_plane[2], self.num_classes)

    def _make_layer(self, feature_n, out_planes, shuffle):
        layers = []
        for i in range(feature_n):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0

            layers.append(StageV2(self.in_planes, out_planes - cat_planes, stride=stride, shuffle=shuffle))
            
            self.in_planes = out_planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.bn5(self.conv5(out))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
        

def test():
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ShuffleNet('ShuffleNet', group=1, shuffle=1, scale=1)
    # model = ShuffleNetV2('ShuffleNetV2', c_size=1, shuffle=0)
    model = model.to(device)
    
    # print(model)

    inp = torch.randn(1, 3, 32, 32).to(device)
 
    torchsummary.summary(model, input_size=(3, 32 ,32))

    macs , params = profile(model, inputs=(inp,))
    print('MACs : ', macs, 'Params : ', params)

if __name__ == '__main__':
    test()
