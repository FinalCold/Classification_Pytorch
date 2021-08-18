import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
num_classes = 10

class VGG(nn.Module):
    def __init__(self, name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[name])
        self.fclayer = nn.Sequential(
                            nn.Linear(512, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, num_classes),
                            # nn.Softmax(dim=1), # Loss인 Cross Entropy Loss 에서 softmax를 포함한다.
                            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fclayer(x)

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers +=  [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers +=  [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                            nn.BatchNorm2d(x),
                            nn.ReLU(inplace=True)]
                            # nn.Dropout(p=0.3)]
                in_channels = x

        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]

        # Passing as an Asterisk argument
        return nn.Sequential(*layers)

def test():
    network = VGG('VGG16')

    print(network)

# test()