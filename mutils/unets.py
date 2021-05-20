from numpy import pi, sqrt, exp, log
import numpy as np
import torch
from torch import cat
from functools import partial
from torch.nn import Module, Sequential, ModuleList, ModuleDict, ReLU, Conv2d, ConvTranspose2d, BatchNorm2d, \
    MaxPool2d, AvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d

class UFLandmarks(Module):
    def __init__(self, color_number, n_landmarks):
        super().__init__()
        self.n_landmarks = n_landmarks
        self.downs = ModuleDict({
            '0': Sequential(Conv2d(color_number, 64, 3, padding=1), ReLU(), Conv2d(64, 64, 3, padding=1), ReLU()),  # 224
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

def bell_5gauss(r):
    sum_5gauss = np.zeros_like(r)
    for s in range(5):
        sigma = 2*s + 1
        sum_5gauss += 2/5*pi * sigma**2 * gauss(r, sigma)
    # sum_5gauss = gauss(r)
    return sum_5gauss

def argmax2d(matrix):
    shape = matrix.shape
    H, W = shape[-2:]
    
    args = torch.argmax(matrix.view(-1, H*W), dim=-1)
    tmp = [(x//W, x%W) for x in args]
    tmp = torch.tensor(tmp)
    args2d = tmp.view(*shape[:-2],2)
    return args2d

    
class LandmarksLoss(Module):
    def __init__(self, mode='round_5gauss', sigma=1., img_size=224):
        super().__init__()
        if type(img_size) is tuple:
            self.img_size = np.array(img_size)
        else:
            self.img_size = np.array((int(img_size), int(img_size)))
        self.delta = 128  # rest of delta x delta shape are nulls
        # self.bell = torch.zeros(self.img_size*2)
        
        if 'round' in mode:
            pass  # Loss uses saved matrix
        else:
            # This loss should use every time evaluation of N(0, sigma)
            assert False, 'Only round modes are supported'
        bell_func = partial(gauss, sigma=sigma)
        if '5gauss' in mode:
            bell_func = bell_5gauss
        self.bell = np.zeros((self.delta, self.delta))
        for x in range(self.delta):
            for y in range(self.delta):
                r = sqrt((x - self.delta/2)**2 + (y - self.delta/2)**2)
                self.bell[x, y] = bell_func(r)
        self.bell = torch.tensor(self.bell)
        # for x in range(self.delta):
            # for y in range(self.delta):
                # self.bell[self.img_size[0]//2 + x, self.img_size[1]//2 + y] = \
                # bell_func(sqrt(x**2 + y**2))
        self.loss = torch.nn.MSELoss()

    def get_heatmap_from(self, marks, shape, bell):
        true_heatmap = torch.zeros(shape)
        batch_size, landmarks_num, H, W = shape
        for img_num, marks in enumerate(marks.view(batch_size, -1)):
            for mark_num, (y, x) in enumerate(marks.view(landmarks_num, 2)):
                x_rounded = torch.round(x).long()
                y_rounded = torch.round(y).long()
                x_left = x_rounded - max(0, x_rounded - bell.shape[0]//2)
                x_right= min(H, x_rounded + bell.shape[0] - bell.shape[0]//2) - x_rounded
                y_high = min(W, y_rounded + bell.shape[1] - bell.shape[1]//2) - y_rounded
                y_low  = y_rounded - max(0, y_rounded - bell.shape[1]//2)
                true_heatmap[img_num, mark_num, x_rounded - x_left:x_rounded + x_right,
                            y_rounded - y_low:y_rounded + y_high] = bell[
                            bell.shape[0]//2 - x_left: bell.shape[0]//2 + x_right,
                            bell.shape[0]//2 - y_low : bell.shape[1]//2 + y_high]
        return true_heatmap
    def forward(self, pred_heatmap, true_landmarks):
        # pred_heatmap.shape == (batch_size, maps_number, H, W)
        # true_landmarks.shape == (maps_number, 2)
        device = pred_heatmap.device
        true_heatmap = self.get_heatmap_from(true_landmarks, pred_heatmap.shape, self.bell).to(device)
        return self.loss(pred_heatmap, true_heatmap)

class DiceLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, gt):  # (batch_size, C, H, W)
        intersection = torch.mean(2 * pred * gt, dim=(-1,-2))
        union = torch.mean(pred * pred + gt * gt, dim=(-1,-2))
        epsillon = 1e-7
        return torch.mean((intersection + epsillon) / (union + epsillon))

s2p = sqrt(2 * pi)
def gauss(r, sigma=1.0, a = 0.):
    a = 1.
    return exp(-((r - a) / (2 * sigma)) ** 2) / s2p

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