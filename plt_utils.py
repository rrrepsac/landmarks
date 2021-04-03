#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:53:49 2021

@author: useren
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.axes import Axes
#from .dlib_utils import ltrb2xywh

def draw_rectangle(rectangle_xy_w_h, ax:Axes=None, edgecolor='blue', facecolor='None'):
    if ax is None:
        ax = plt.gca()
    ax.add_patch(Rectangle(*rectangle_xy_w_h, edgecolor=edgecolor, facecolor=facecolor))

def draw_marks(marks, ax:Axes=None, color='red', xy=True):
    if ax is None:
        ax = plt.gca()
    # marks_q = len(marks) // 2
    if xy:
        ax.scatter(marks[0::2], marks[1::2], color=color)
    else:
        ax.scatter(marks[1::2], marks[0::2], color=color)
    