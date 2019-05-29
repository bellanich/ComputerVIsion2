import numpy as np
import torch
import torchgeometry as tgm
from math import sin, cos, tan, pi
from mesh_to_png import load_faces, mesh_to_png
from data_def import Mesh


def get_viewport_matrix(vr, vl, vt, vb):
    """
    create a viewport matrix that is defined by the coordinates (vl, vt) and (vr, vb)
    :param vl:
    :param vr:
    :param vt:
    :param vb:
    :return:
    """
    return torch.tensor([[(vr - vl)/2, 0,    0, (vr + vl) / 2],
                    [0,  (vt - vb)/2,    0, (vt + vb) / 2],
                    [0,            0,  0.5,           0.5],
                    [0,            0,    0,             1]], dtype=torch.float64)


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
    return torch.tensor([[2*n / (r - l),                     0,                  0, 0],
                     [0,                     2*n / (t - b),                  0, 0],
                     [(r + l) / (r - l), (t + b) / (t - b), (-f - n) / (f - n), -1],
                     [0,                                 0,   -2*f*n / (f - n), 0]], dtype=torch.float64)


def get_transformation_matrix(angles, translation):
    """
    Create a rotation matrix defined by the rotation angles around the x, y and z axes
    :param angles:
    :param translation:
    :return:
    """
    rtvec = torch.cat((tgm.deg2rad(angles), translation), 0)
    rtvec = torch.unsqueeze(rtvec, 0)

    transformation_matrix = tgm.rtvec_to_pose(rtvec)

    return transformation_matrix

    # degree_to_radians = 2 * pi / 360.
    # angles = angles * degree_to_radians
    #
    # Rx = np.array([[1,              0,               0],
    #                [0, cos(angles[0]), -sin(angles[0])],
    #                [0, sin(angles[0]),  cos(angles[0])]])
    #
    # Ry = np.array([[cos(angles[1]),  0, sin(angles[1])],
    #                [0,               1,              0],
    #                [-sin(angles[1]), 0, cos(angles[1])]])
    #
    # Rz = np.array([[cos(angles[2]), -sin(angles[2]), 0],
    #                [sin(angles[2]),  cos(angles[2]), 0],
    #                [0,                            0, 1]])
    #
    # rotation = np.matmul(np.matmul(Rx,Ry),Rz)
    #
    # return np.append(np.append(rotation, np.zeros((1, 3)), axis=0), np.array([[translation[0]], [translation[1]],
    #                                                                           [translation[2]], [1]]), axis=1)


def load_txt(filename):
    """
    load an .txt file into a list
    :param filename:
    :return:
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    landmarks = np.array([int(line.rstrip()) for line in lines])

    return landmarks


def run_pinhole_camera(angles, translation, pCexp, mean_tex, use_landmarks=False, test=False):

    fovy = 0.5
    top = 1.
    right = top
    left = bottom = - top
    near = top / tan(fovy / 2)
    far = 100

    # Add fourth dimension to points
    pCexp = torch.cat((pCexp, torch.ones((len(pCexp)), 1, dtype=torch.float64)), 1)

    # Projection matrix
    V = get_viewport_matrix(1, -10, 1, -10)
    PP = get_perspective_projection_matrix(1, -1, 1, -1, 100, 0.5)
    # P = np.matmul(V, PP)
    P = V * PP

    # Transformation matrix
    T = get_transformation_matrix(angles, translation)

    full_transformation_matrix = torch.squeeze(torch.matmul(P, T), 0)
    pCexp = torch.transpose(pCexp, 0, 1)
    transformed_pCexp = torch.transpose(torch.matmul(full_transformation_matrix, pCexp), 0, 1)

    # transformed_points = np.empty((0, 4))
    # for i in range(len(pCexp)):
    #     transposed_point = np.expand_dims(pCexp[i], 1)
    #     transformed_point = np.matmul(np.matmul(P, T), transposed_point)
    #
    #     # transformed_point[0] = transformed_point[0] * transformed_point[2] / transformed_point[3]
    #     # transformed_point[1] = transformed_point[1] * transformed_point[2] / transformed_point[3]
    #     # transformed_point[2] = transformed_point[2] / transformed_point[3]
    #
    #     transformed_points = np.append(transformed_points, transformed_point.T, axis=0)

    if use_landmarks:
        landmark_indices = load_txt('./Data/Landmarks68_model2017-1_face12_nomouth.txt')

        landmarks = transformed_pCexp[landmark_indices][:, :2]

        for i in range(len(landmark_indices)):
            mean_tex[landmark_indices[i]] = [0, 0, 1]
    else:
        landmarks = None

    return transformed_pCexp, landmarks


if __name__ == "__main__":

    # because of rendering problem I needed to run this program for every mesh_to_png seperately

    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    pCid, pCexp, mean_tex, triangles = load_faces(alpha, delta)
    mesh = Mesh(pCexp, mean_tex, triangles)
    mesh_to_png('pinhole__.png', mesh)

    translation = np.array([0, 0, 0])

    # -10 degrees around y axis
    transformed_face, _ = run_pinhole_camera(torch.tensor([0, -10, 0]), translation, pCexp, mean_tex, test=True)
    mesh = Mesh(transformed_face[:, :-1], mean_tex, triangles)
    mesh_to_png('pinhole_-10__.png', mesh)

    # +10 degrees around y axis
    transformed_face, _ = run_pinhole_camera(torch.tensor([0, 10, 0]), translation, pCexp, mean_tex, test=True)
    mesh = Mesh(transformed_face[:, :-1], mean_tex, triangles)
    mesh_to_png('pinhole_10__.png', mesh)

    # +10 degrees with landmarks
    transformed_face, _ = run_pinhole_camera(torch.tensor([0, 10, 0]), translation, pCexp, mean_tex, use_landmarks=True, test=True)
    mesh = Mesh(transformed_face[:,:-1], mean_tex, triangles)
    mesh_to_png('pinhole_10_with_landmarks__.png', mesh)
    
