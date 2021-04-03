from numpy import pi, sin, cos, array, arctan2, sqrt
import numpy as np

def perspective_marks(marks, coeffs):

# """
# The 8 transform coefficients (a, b, c, d, e, f, g, h) correspond to the following transformation:
# 
# x' = (ax + by + c) / (gx + hy + 1)
# y' = (dx + ey + f) / (gx + hy + 1)
# """
    a, b, c, d, e, f, g, h = coeffs
    x, y = array(marks).transpose((-1, -2))
    x_persp = (a*x + b*y + c) / (g*x + h*y + 1)
    y_persp = (d*x + e*y + f) / (g*x + h*y + 1)
    return np.vstack((x_persp, y_persp)).transpose((-1, -2)).astype(np.float32)

def find_perspective_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def decart2polar(marks, xy=True):
    # marks must be np.array-x, np.array-y
    # marks = array(marks)
    x, y = marks#.transpose()
    if not xy:
        x, y = y, x
    r, a = sqrt(x**2 + y**2), arctan2(y, x)
    polar = np.vstack((r, a))
    return polar

    
def polar2decart(polar_marks, xy=True):
    # marks must be np.array-x, np.array-y
    # marks = array(marks)
    r, a = polar_marks
    x, y = r * cos(a), r * sin(a)
    if not xy:
        x, y = y, x
    decart_marks = np.vstack((x, y))
    return decart_marks

    
def rotate_marks(marks, degree, center=(0, 0), xy=True):
    degree = degree / 180 * pi
    # centred_marks = array(centred_marks).transpose((-1,-2))
    marks = array(marks).transpose((-1,-2))
    center = array([center]).transpose()
    centred_marks = marks - center
    
    # print(marks)
    # c = array(img_size)/2
    # assert False
    # print(np.shape(marks), np.shape(center))
    polar_marks = decart2polar(centred_marks, xy)
    polar_marks[1] += degree
    rotated_decart_marks = polar2decart(polar_marks, xy)
    decentred_rotated_decart_marks = rotated_decart_marks + center
    rotated_marks = decentred_rotated_decart_marks.transpose().astype(np.float32)#.tolist()
    # print(rotated_marks)
    return rotated_marks

