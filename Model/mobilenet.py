import torch
import torch.nn as nn
import math
import numpy as np
import torchsummary
from thop import profile


def _make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

def conv_3x3_bn(inp, outp, stride):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 3, stride, 1, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, outp):
    return nn.Sequential(
        nn.Conv2d(inp, outp, 1, 1, 0, bias=False),
        nn.BatchNorm2d(outp),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, outp, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp, expand_ratio)
        if stride == 1 and inp == outp:
            self.identity = True
        else:
            self.identity = False

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw_linear
                nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outp),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw_linear
                nn.Conv2d(hidden_dim, outp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(outp),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.cfgs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        block = InvertedResidual

        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0  else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileNetV2()
    model = model.to(device)

    inp = torch.randn(1, 3, 32, 32).to(device)
 
    torchsummary.summary(model, input_size=(3, 32 ,32))

    macs , params = profile(model, inputs=(inp,))
    print('MACs : ', macs, 'Params : ', params)

if __name__ == '__main__':
    test()