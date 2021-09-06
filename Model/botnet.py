import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from thop import profile

cfg = {
    'BOT50': ['Bottleneck', [3, 4, 6, 3]],
    'BOT101': ['Bottleneck', [3, 4, 23, 3]],
    'BOT152': ['Bottleneck', [3, 8, 36, 3]],
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, heads=4, mhsa=False, resolution=(32, 32)):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Normal Layer
        if not mhsa :
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        
        # MHSA Layer
        else:
            self.conv2 = nn.ModuleList()
            self.conv2.append(MHSA(planes, width=resolution[0], height=resolution[1], heads=heads))
            if stride == 2:
                self.conv2.append(nn.AvgPool2d(2, 2))
            self.conv2 = nn.Sequential(*self.conv2)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class MHSA(nn.Module):
    def __init__(self, n_dims, width, height, heads):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        # print(x.size())
        # print(content_content.size(), content_position.size())
        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

class BoTNet(nn.Module):
    def __init__(self, name, num_classes = 10, resolution=(32, 32)):
        super(BoTNet, self).__init__()
        self.in_planes = 64
        self.num_classes = num_classes
        self.resolution = list(resolution)

        self.block_name = eval(cfg[name][0])
        self.block_feature = cfg[name][1]
        
        # print(self.block_name, self.block_feature)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(self.block_name, self.block_feature[0], 64, stride=1)
        self.layer2 = self._make_layer(self.block_name, self.block_feature[1], 128, stride=2)
        self.layer3 = self._make_layer(self.block_name, self.block_feature[2], 256, stride=2)
        self.layer4 = self._make_layer(self.block_name, self.block_feature[3], 512, stride=2, heads=4, mhsa=True)
        self.linear = nn.Linear(512*self.block_name.expansion, self.num_classes)

    def _make_layer(self, block_name, feature, planes, stride, heads=4, mhsa=False):
        strides = [stride] + [1]*(feature-1)
        layers = []
        for stride in strides:
            layers.append(block_name(self.in_planes, planes, stride, heads=heads, mhsa=mhsa, resolution=self.resolution))
            if stride == 2:
                # print(self.resolution)
                self.resolution[:] = [int(x / 2) for x in self.resolution]
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

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BoTNet('BOT50')
    model = model.to(device)

    inp = torch.randn(1, 3, 32, 32).to(device)
 
    torchsummary.summary(model, input_size=(3, 32 ,32))

    macs , params = profile(model, inputs=(inp,))
    print('MACs : ', macs, 'Params : ', params)

if __name__ == '__main__':
    test()