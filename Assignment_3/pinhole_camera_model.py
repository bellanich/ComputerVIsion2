import mesh_to_png
import numpy as np
from math import sin, cos, pi


def get_viewport_matrix(vl, vr, vt, vb):
    """
    create a viewport matrix that is defined by the coordinates (vl, vt) and (vr, vb)
    :param vl:
    :param vr:
    :param vt:
    :param vb:
    :return:
    """
    return np.array([[(vr - vl)/2, 0,  0,   0],
                    [0,  (vt - vb)/2,  0,   0],
                    [0,            0,  0.5, 0],
                    [0,            0,  0,   1]])


def get_perspective_projection_matrix(r, l, t, b, f, n):
    """
    create a perspective projective matrix defined by the parameters
    :param r:
    :param l:
    :param t:
    :param b:
    :param f:
    :return:
    """
    return np.array([[2*n / (r - l), 0,  (r + l) / (r - l),                0],
                     [0, 2*n / (t - b),  (t + b) / (t - b),                0],
                     [0,             0, (-f - n) / (f - n), -2*f*n / (f - n)],
                     [0,             0,                 -1,                0]])


def get_rotation_matrix(angles):
    """
    Create a rotation matrix defined by the rotation angles around the x, y and z axes
    :param angels:
    :return:
    """
    degree_to_radians = 2 * pi / 360
    angles = angles * degree_to_radians

    Rx = np.array([[1,              0,               0],
                   [0, cos(angles[0]), -sin(angles[0])],
                   [0, sin(angles[0]),  cos(angles[0])]])

    Ry = np.array([[cos(angles[1]),  0, sin(angles[1])],
                   [0,               1,              0],
                   [-sin(angles[1]), 0, cos(angles[1])]])

    Rz = np.array([[cos(angles[2]), -sin(angles[2]), 0],
                   [sin(angles[2]),  cos(angles[2]), 0],
                   [0,                            0, 1]])

    return Rx*Ry*Rz
