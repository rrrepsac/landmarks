import torch
from torch import cat
from torch.nn import Module, Sequential, ModuleList, ModuleDict, ReLU, Conv2d, ConvTranspose2d, BatchNorm2d, \
    MaxPool2d, AvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d

class UFLandmarks(Module):
    def __init__(self, n_landmarks):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.downs = ModuleDict({
            '0': Sequential(Conv2d(3, 64, 3, padding=1), ReLU(), Conv2d(64, 64, 3, padding=1), ReLU()),  # 224
            'pool2': MaxPool2d((2, 2)),
            '1': Sequential(Conv2d(64, 128, 3, padding=1), ReLU(), Conv2d(128, 128, 3, padding=1), ReLU()), # 112
            # MaxPool2d((2, 2)),
            '2': Sequential(Conv2d(128, 256, 3, padding=1), ReLU(), Conv2d(256, 256, 3, padding=1), ReLU()), # 56
            # MaxPool2d((2, 2)),
            '3': Sequential(Conv2d(256, 512, 3, padding=1), ReLU(), Conv2d(512, 512, 3, padding=1), ReLU()), # 28
        })
        self.ups = ModuleDict({
            'upconv2_0': ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            '0': Sequential(Conv2d(512, 256, 3, padding=1), ReLU(), Conv2d(256, 256, 3, padding=1), ReLU()),  # 224
            'upconv2_1': ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            '1': Sequential(Conv2d(256, 128, 3, padding=1), ReLU(), Conv2d(128, 128, 3, padding=1), ReLU()), # 112
            'upconv2_2': ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            '2': Sequential(Conv2d(128, 64, 3, padding=1), ReLU(), Conv2d(64, 64, 3, padding=1), ReLU()), # 56
            'out': Conv2d(64, n_landmarks, 1)
        })

    def forward(self, x):
        down_out = []
        for di in range(3):
            x = self.downs[str(di)](x)
            down_out.append(x)
            x = self.downs['pool2'](x)
        x = self.downs[str(3)](x)
        for ui in range(3):
            upconv = f'upconv2_{ui}'
            x = self.ups[upconv](x)
            x = cat((x, down_out[2 - ui]), dim=1)
            x = self.ups[str(ui)](x)
        heatmaps = self.ups['out'](x)
        return heatmaps

###
if __name__ == '__main__':
    print('main')
    ufl = UFLandmarks(1)
    x = torch.randn(1,3,224,224)
    y = ufl(x)