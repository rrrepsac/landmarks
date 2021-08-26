#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:22:42 2021

@author: honor
"""

from .unets import bell_max, get_heatmap_from
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, norm
import torch
from torch.utils.data import Dataset, DataLoader

from . import plt_utils
from . import dlib_utils

from . import transform_utils


# =============================================================================
#  inverse_modes_mapping = {
#         0: InterpolationMode.NEAREST,
#         2: InterpolationMode.BILINEAR,
#         3: InterpolationMode.BICUBIC,
#         4: InterpolationMode.BOX,
#         5: InterpolationMode.HAMMING,
#         1: InterpolationMode.LANCZOS,
#     }
# =============================================================================
# from zipfile import ZipFile
from shutil import unpack_archive, rmtree
from pathlib import Path
import json
import shutil
from torch.nn import Module, Sequential, Conv2d, MaxPool2d, AdaptiveMaxPool2d,\
    BatchNorm2d, Dropout, ReLU, Tanh, Sigmoid, Linear
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomAffine, RandomPerspective
import matplotlib

from mutils.unets import LandmarksLoss, UFLandmarks, argmax2d

def nan_to_num(matrix):
    matrix[matrix.isnan()] = 0
    return matrix

class CNN_landmarks(Module):
    def __init__(self, landmarks_number, resize):
        print(matplotlib.__version__, 'plt')
        super().__init__()
        self.bn = BatchNorm2d(3)
        self.conv1 = Sequential(Conv2d(3, 32, 3, padding=1),  # BatchNorm2d(32),
                                Tanh(), )
        self.maxpool = MaxPool2d((2*8//resize, 2*8//resize))
        # self.drop1 = Dropout(0.25)
        # self.fc1 = Sequential(Linear(32 * 32 * 32, 256))
        # self.drop2 = Dropout(0.5)
        # self.fc2 = Sequential(Linear(256, landmarks_number * 2))  # , Sigmoid())
        self.fc = Linear(32*32*32, landmarks_number*2)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        # x = self.drop1(x)
        # x = self.fc1(x.view(x.size()[0], -1))
        # x = self.drop2(x)
        # x = self.fc2(x)
        return self.fc(x.view(x.size()[0], -1))

class FaceLandmarks(Dataset):
    def __init__(self, path: Path, landmarks_number=4, mode='PIL', resize=8, distortion_scale=0.5, max_affine_degree=120, new_size=None):
        # 33 -- nose, 41 -- left eye, 44 -- right eye, 62 -- mouth
        landmarks_list = [41, 44, 33, 62]
        self.landmarks_list = landmarks_list[: landmarks_number]
        self.distortion_scale = distortion_scale
        self.max_affine_degree = max_affine_degree
        self.resize = resize
        self.new_size = new_size
        if self.new_size and type(self.new_size) is not tuple:
            self.new_size = (self.new_size, self.new_size)
        super().__init__()
        self.path = Path('./')
        self.del_path = False
        # refactor with #In case none is found, a ValueError is raised.
        if str(path).endswith('.zip') and not Path('./all_data.json').exists():
            self.del_path = True
            unpack_archive(path)
            self.path = Path('./')
        with open('all_data.json') as json_ref:
            self.all_data = json.loads(json_ref.read())
        self.mode = mode
        self.get_rand_affine_params = RandomAffine.get_params
        self.get_rand_perspective_params = RandomPerspective.get_params
        self.persp_params = []

    def __len__(self):
        return len(self.all_data.keys())

    # def __del__(self):
        # if self.del_path:
        # rmtree(self.path/'images')
        # rmtree(self.path/'all_data.json')

    def __getitem__(self, index):
        # print(self.all_data[str(index)])
        # print(self.all_data[str(index)]['file_name'])
        # print(self.all_data[index])
        img_path = self.path/'images'/self.all_data[str(index)]['file_name']
        img = Image.open(img_path, 'r')
        size = img.size
        if not self.new_size:
            self.new_size = (size[0] // self.resize, size[1] // self.resize)
        img = img.resize(self.new_size)
        # new_size = img.size
        face_landmarks = [
            self.all_data[str(index)]['face_landmarks'][i] for i in self.landmarks_list]
        # face_landmarks = self.all_data[str(index)]['face_landmarks']
        if self.mode == 'tensor':
            img = ToTensor()(img)
            face_landmarks = torch.Tensor(face_landmarks) / self.resize
            if self.max_affine_degree > 0:
                affine_inerval = [-self.max_affine_degree, self.max_affine_degree]
                affine_params = self.get_rand_affine_params(affine_inerval, None, None, None, img_size=self.new_size)
                img = TF.affine(img, *affine_params, resample=2)
                face_landmarks = transform_utils.rotate_marks(face_landmarks, affine_params[0], array(self.new_size)/2)
            # print('p_params', persp_params)
            # print('aff', face_landmarks)
            if self.distortion_scale > 0:
                self.persp_params = self.get_rand_perspective_params(width=self.new_size[0], height=self.new_size[1], distortion_scale=self.distortion_scale)
                # self.persp_params = torch.Tensor(
                    # [[[0.0, 0.0],  [31.0, 0.0], [31.0, 31.0], [0.0, 31.0]],
                    # [[10.0, 13.0], [23.0, 0.0], [22.0, 29.0], [3.0, 20.0]]])/(self.resize/16)
                persp_coeffs = transform_utils.find_perspective_coeffs(*self.persp_params)
                img = TF.perspective(img, *self.persp_params)
                face_landmarks = transform_utils.perspective_marks(face_landmarks, persp_coeffs)
            # print('persp', face_landmarks)
        img = nan_to_num(img)
        # face_landmarks = nan_to_num(face_landmarks)
        return img, face_landmarks

def test_cnn(resize=4, landmarks_number=3, batch_size=8, epochs=5, shuffle=True, max_affine_deg=120, distortion_scale=0.8):
    print(torchvision.__version__)
    
    
    cnn = CNN_landmarks(landmarks_number, resize)
    ds = FaceLandmarks(Path('./face_landmarks.zip'),
                       landmarks_number, mode='tensor', resize=resize,
                       distortion_scale=distortion_scale, max_affine_degree=max_affine_deg)
    optim = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        for bn, (batch_img, batch_landmarks) in enumerate(DataLoader(ds, batch_size=batch_size, shuffle=shuffle)):
            # max = torch.max(batch_landmarks.abs())
            # if max is np.nan or max > 127:
                # print(bn, batch_landmarks)
            # continue
            # print(batch_img.shape[-2:])
            # params = torchvision.transforms.RandomAffine.get_params([-120,120], None, None, None, img_size=batch_img.shape[-2:])
            # print(params)
            # assert False
            # img0 = ToPILImage()(TF.affine(batch_img[0].unsqueeze(0), *params, resample=2,)[0])
            # img0 = ToPILImage()(batch_img[0])
            # plt.imshow(img0)
            # m2 = batch_landmarks[0]
            # marks_r = transform_utils.rotate_marks(batch_landmarks[0], img0.size, params[0])
            # print(ToPILImage()(batch_img[0]).size)
            # print(batch_landmarks[0])
            # print(marks_r.reshape(-1))
            # plt_utils.draw_marks(batch_landmarks[0].reshape(-1), color='r', xy=False)
            # plt.show()
            # return None
            # assert False
            # print(batch_landmarks)
            # assert False
            optim.zero_grad()
            if False:
                describe_state({'img':batch_img})
                with open('err.log', 'w') as flog:
                    for m in batch_img[0]:
                        for rw in m:
                            for el in rw:
                                print(f'{np.float(el):4.1f}', file=flog, end=' ')
                            print(file=flog)
                        print(file=flog)
                    # print(batch_img, file=flog)
            pred_landmarks = cnn(batch_img)
            loss = criterion(batch_landmarks.view(pred_landmarks.shape), pred_landmarks)
            loss.backward()
            optim.step()
            # if loss != loss:
                # print('naaaaan')
            if bn % 80 == 0 or loss != loss:
                print(
                    f'{epoch}_{epoch*len(ds) + bn*batch_size:5d} img passed loss = {loss:.3e}')
                img = batch_img[0].detach().unsqueeze(0)
                # print(trans1.transforms[1].get_params(*img.shape[2:], 0.6))
                # print(trans1.transforms[1].get_params(*img.shape[2:], 0.6))
                # img = trans1(img)
                # print(*img.shape[2:])
                with torch.no_grad():
                    pred_test = cnn(img.detach()).detach()
                pil_img = ToPILImage()(img[0])    
                plt.imshow(pil_img)
                # print(pred_landmarks.shape)
                # landmarks_xy = list(
                # zip(*pred_test.detach()[0].view(-1, 2)))
                # landmarks_xy = list(zip(*(pred_landmarks[0])))

                # plt.scatter(landmarks_xy[0], landmarks_xy[1])
                # print(pred_test)
                # print(batch_landmarks[0].view(-1))
                # describe_state(cnn.state_dict())
                plt_utils.draw_marks(pred_test.detach()[0], color='g')
                plt_utils.draw_marks(batch_landmarks.detach()[0].view(-1), color='r')
                dlib_utils.draw_landmarks(pil_img, color='blue')
                # assert False
                

                plt.show()
                # if loss != loss:
                    # print(torch.Tensor(pp).tolist())
                    # assert False

    return None

def test_unet(resize=4, landmarks_number=3, batch_size=8, epochs=5, shuffle=True,
              max_affine_deg=0, distortion_scale=0.):
    print(torchvision.__version__)
    
    
    model = UFLandmarks(3, landmarks_number)

    ds = FaceLandmarks(Path('./face_landmarks.zip'),
                       landmarks_number, mode='tensor', resize=resize, #new_size=96,
                       distortion_scale=distortion_scale, max_affine_degree=max_affine_deg)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = LandmarksLoss('round_gauss', sigma=4.)

    
    for epoch in range(epochs):
        for bn, (batch_img, batch_landmarks) in enumerate(DataLoader(ds, batch_size=batch_size, shuffle=shuffle)):
            optim.zero_grad()

            pred_heatmap = model(batch_img)
            loss = criterion(pred_heatmap, batch_landmarks.view(-1,2))
            loss.backward()
            optim.step()
            # if loss != loss:
                # print('naaaaan')
            if bn % 10 == 0 or loss != loss:
                print(
                    f'{epoch}_{epoch*len(ds) + bn*batch_size:5d} img passed loss = {loss:.3e}')
                img = batch_img[0].detach().unsqueeze(0)
                with torch.no_grad():
                    pred_test = model(img.detach()).detach()
                pred_marks = argmax2d(pred_test)#, dim=(-2, -1))
                pil_img = ToPILImage()(img[0])
                ax = plt.subplot(131)   
                ax.imshow(pil_img)
                ax = plt.subplot(132)
                true_heatmap = get_heatmap_from(batch_landmarks.detach().view(-1),
                                                        pred_heatmap.shape, criterion.bell)
                sum_heatmap = torch.zeros_like(pred_heatmap[0][0])
                for mn in range(landmarks_number):
                    sum_heatmap = sum_heatmap + pred_heatmap[0][mn]
                    break
                arg_pred = bell_max(sum_heatmap, criterion.bell)
                ax.imshow(ToPILImage()(sum_heatmap), alpha=0.5)
                print(arg_pred, ' -- predict')
                ax.scatter(arg_pred[0], arg_pred[1], color='green')
                # plt_utils.draw_marks(pred_marks[0].view(-1), color='r')
                ax = plt.subplot(133)
                sum_heatmap = torch.zeros_like(true_heatmap[0][0])
                for mn in range(landmarks_number):
                    sum_heatmap = sum_heatmap + true_heatmap[0][mn]
                    break
                ax.imshow(ToPILImage()(sum_heatmap), alpha=1.)
                arg_pred = bell_max(sum_heatmap, criterion.bell)
                print(arg_pred, ' -- true')
                ax.scatter(arg_pred[0], arg_pred[1], color='red')
                # plt_utils.draw_marks(batch_landmarks.detach()[0].view(-1), color='g')
                # dlib_utils.draw_landmarks(pil_img, color='blue')
                # assert False
                

                plt.show()
                # if loss != loss:
                    # print(torch.Tensor(pp).tolist())
                    # assert False

    return None

def describe_state(state_dict):
    print(state_dict.keys())
    for key, val in state_dict.items():
        print(key, type(val), val.dtype)
        # continue
        if type(val) is torch.Tensor and val.dtype is torch.float:
            print(key, end=': ')
            std, mean = torch.std_mean(val)
            abs_max = val.abs().max()
            print(f'abs_max={abs_max}, std={std}, mean={mean}')

print(__name__)
from numpy import array
if __name__ == '__main__':
    test_cnn()
# test2(Image.open('2.jpg'))

