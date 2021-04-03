#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:51:43 2021

@author: useren
"""

import dlib
import numpy as np
from .plt_utils import draw_rectangle, draw_marks
from os.path import isfile
from urllib.request import urlretrieve
import bz2
#from PIL import Image

face_detect = None
predict_5_face_landmarks = None
predict_68_face_landmarks= None

def ltrb2xywh(rect: dlib.rectangle):
  l, t, r, b = rect.left(), rect.top(), rect.right(), rect.bottom()
  return (l, t), r - l, b - t

def get_np_4dlib(image_PIL):
    image_np = np.array(image_PIL)
    image_shape = image_np.shape
    if len(image_shape) == 4:  # squeeze(0)
        if image_shape[0] > 1:
            print('Only one image per call accepts.')
            return None
        image_shape = image_shape[1:]
        image_np = image_np.reshape(image_shape)
    if len(image_shape) == 3 and image_shape[-1] == 1:  # squeeze(-1), because
                                    # dlib gets (HxW) array for grayscale image
        image_shape = image_shape[:-1]
        image_np = image_np.reshape(image_shape)
    return image_np

def draw_landmarks(image_PIL, landmarks_number=5, color='red'):
    image_np = get_np_4dlib(image_PIL)
    marks = get_landmarks(image_np, landmarks_number)
    if  marks:
        draw_marks(marks, color=color)
    return marks

def get_landmarks(image_np, landmarks_number=5):
    global predict_5_face_landmarks
    global predict_68_face_landmarks
    if landmarks_number not in (5, 68):
        print('Wrong landmarks number. Choose 5 or 68. Using 5 landmarks now!')
        landmarks_number = 5
    predictor = (predict_5_face_landmarks, predict_68_face_landmarks)[landmarks_number == 68]
    if predictor is None:
        aux_dlib_url = 'https://github.com/davisking/dlib-models/raw/master/'
        # aux_dlib_url = 'https://github.com/davisking/dlib-models/blob/master/'  # shape_predictor_5_face_landmarks.dat.bz2
        aux_predictor_files = ['shape_predictor_5_face_landmarks.dat', 'shape_predictor_68_face_landmarks.dat']
        aux_file = aux_predictor_files[landmarks_number == 68]
        if not isfile(aux_file):
            if not isfile(aux_file + '.bz2'):
                retr_ret = urlretrieve(aux_dlib_url + aux_file + '.bz2', aux_file + '.bz2')
            dec_aux = bz2.BZ2File(aux_file + '.bz2').read()
            with open(aux_file, 'wb') as f:
                f.write(dec_aux)
            #print(retr_ret)
        #print(predictor, (predict_5_face_landmarks, predict_68_face_landmarks), isfile(aux_file))
        predictor = dlib.shape_predictor(aux_file)
        if landmarks_number == 5:
            predict_5_face_landmarks = predictor
        else:
            predict_68_face_landmarks= predictor
        #print(predictor, (predict_5_face_landmarks, predict_68_face_landmarks))
    faces_rectangles = detect_faces(image_np)
    if len(faces_rectangles) == 0:
        image_shape = image_np.shape
        faces_rectangles = [dlib.rectangle(left=0, top=0, right=image_shape[0], bottom=image_shape[1])]
    for face_rectangle in faces_rectangles:
        shape = predictor(image_np, face_rectangle)
        #print(shape.part)
        #print(shape.parts)
        marks = [0]*2*shape.num_parts
        #print(shape.num_parts, '=marks_num')
        for i in range(shape.num_parts):
            marks[2*i: 2*i + 2] = shape.part(i).x, shape.part(i).y
        #print(marks)
        return marks #do only first face

    return None

def detect_faces(image_np):
    global face_detect
    if face_detect is None:
        face_detect = dlib.get_frontal_face_detector()

    return face_detect(image_np)

def draw_faces(image_PIL):
    image_np = get_np_4dlib(image_PIL)
    if image_np is None:
        return None
    faces_rectangles = detect_faces(image_np)
    for face_rectangle in faces_rectangles:
        draw_rectangle(ltrb2xywh(face_rectangle))