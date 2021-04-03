#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:22:42 2021

@author: honor
"""

import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, norm
import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage,\
     RandomAffine, RandomPerspective
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
from torchvision.transforms import functional as TF
from torchvision.transforms import RandomAffine, RandomPerspective

class CNN_landmarks(Module):
    def __init__(self, landmarks_number, resize):
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
    def __init__(self, path: Path, landmarks_number=4, mode='PIL', resize=8):
        # 33 -- nose, 41 -- left eye, 44 -- right eye, 62 -- mouth
        landmarks_list = [41, 44, 33, 62]
        self.landmarks_list = landmarks_list[: landmarks_number]
        self.resize = resize
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
        img = img.resize((size[0] // self.resize, size[1] // self.resize))
        new_size = img.size
        face_landmarks = [
            self.all_data[str(index)]['face_landmarks'][i] for i in self.landmarks_list]
        # face_landmarks = self.all_data[str(index)]['face_landmarks']
        if self.mode == 'tensor':
            img = ToTensor()(img)
            face_landmarks = torch.tensor(face_landmarks) / self.resize
            max_affine_degree = 120
            affine_inerval = [-max_affine_degree, max_affine_degree]
            affine_params = self.get_rand_affine_params(affine_inerval, None, None, None, img_size=new_size)
            persp_params = self.get_rand_perspective_params(width=new_size[0], height=new_size[1], distortion_scale=0.8)
            persp_coeffs = transform_utils.find_perspective_coeffs(*persp_params)
            img = TF.affine(img, *affine_params, resample=2)
            img = TF.perspective(img, *persp_params)
            face_landmarks = transform_utils.rotate_marks(face_landmarks, affine_params[0], array(new_size)/2)
            # print('p_params', persp_params)
            # print('aff', face_landmarks)
            face_landmarks = transform_utils.perspective_marks(face_landmarks, persp_coeffs)
            # print('persp', face_landmarks)
            
        return img, face_landmarks

def test_cnn(resize = 4, landmarks_number = 3, batch_size = 8):
    print(torchvision.__version__)
    
    
    cnn = CNN_landmarks(landmarks_number, resize)
    ds = FaceLandmarks(Path('./face_landmarks.zip'),
                       landmarks_number, mode='tensor', resize=resize)
    optim = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(5):
        for bn, (batch_img, batch_landmarks) in enumerate(DataLoader(ds, batch_size=batch_size, shuffle=True)):
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
            pred_landmarks = cnn(batch_img)
            loss = criterion(batch_landmarks.view(
                pred_landmarks.shape), pred_landmarks)
            loss.backward()
            optim.step()
            if bn % 40 == 0:
                print(
                    f'{epoch*len(ds) + bn*batch_size:5d} img passed loss = {loss:.3e}')
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
                describe_state(cnn.state_dict())
                plt_utils.draw_marks(pred_test.detach()[0], color='g')
                plt_utils.draw_marks(batch_landmarks.detach()[0].view(-1), color='r')
                dlib_utils.draw_landmarks(pil_img, color='blue')
                # assert False
                

                plt.show()

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

