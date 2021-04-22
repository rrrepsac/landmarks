from numpy import pi, sqrt, exp, log
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
class DiceLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):  # (batch_size, C, H, W)
        intersection = torch.mean(2 * pred * gt, dim=(-1,-2))
        union = torch.mean(pred * pred + gt * gt, dim=(-1,-2))
        epsillon = 1e-7
        return torch.mean((intersection + epsillon) / (union + epsillon))

s2p = sqrt(2 * pi)
def gauss(x, sigma=1.0, a = 0.):
    a = 1.
    return exp(-((x - a) / (2 * sigma)) ** 2) / s2p

def make_gauss_landmarks(img_size, landmarks, sigma=1.):
    marks_number = landmarks.shape[0]
    ground_truth = torch.zeros((1, marks_number, *img_size))
    delta = int(sqrt(-log(s2p/(2 * 1024*100)))*sigma)
    if delta > img_size[0] // 2:
        delta = img_size[0] // 2
    if delta < 1:
        delta = 1
    print(delta)
    for mark in range(marks_number):
        y, x = landmarks[mark]
        for _i in range(2 * delta):  # dummy way
            for _j in range(2*delta):
                i = int(x - delta + _i)
                if i < 0 or i >= img_size[0]:
                    continue
                j = int(y - delta + _j)
                if j < 0 or j >= img_size[1]:
                    continue
                r = sqrt((x - i)**2 + (y - j)**2)
                fr = gauss(r, sigma)
                ground_truth[0, mark, i, j] = fr
    return ground_truth