from pinhole_camera_model import run_pinhole_camera
from mesh_to_png import load_faces
from face_landmark_detection import face_landmark_detection

import numpy as np


def generate_face(alpha, delta, omega, t):
    """
    :param alpha: a python list
    :param delta: a python list
    :param omega: a python list
    :param t: a python list
    :return:
    """
    pCid, pCexp, mean_tex, triangles = load_faces(alpha, delta)
    transformed_face, landmarks = run_pinhole_camera(omega, t, pCexp, mean_tex, use_landmarks=True)

    return landmarks


def calculate_loss(landmarks, ground_truth_landmarks, alpha, delta, L_alpha, L_delta):
    """
    Calculate the loss function of an ensemble of landmarks with respect to the ground truth landmarks
    :param landmarks: a numpy array
    :param ground_truth_landmarks: a numpy array
    :param alpha: a python list
    :param delta: a python list
    :param L_alpha: a float
    :param L_delta: a float
    :return:
    TODO: expand the loss function to compare 2D vectors in stead of scalar values.
    """
    landmark_loss = 0
    for i in range(0,68):
        landmark_loss += (landmarks[i] - ground_truth_landmarks[i]) ** 2

    alpha_regularization_loss = 0
    for i in range(0,30):
        alpha_regularization_loss += L_alpha * alpha[i] ** 2

    delta_regularization_loss = 0
    for i in range(0,20):
        delta_regularization_loss += L_delta * delta[i] ** 2

    return landmark_loss + alpha_regularization_loss, delta_regularization_loss


def run_update_loop():
    """Use tensorflow as described in the assignment
    TODO: convert ground_truth_landmarks into XY-coordinates
    """
    ground_truth_landmarks = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")
    pass


if __name__ == "__main__":

    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    landmarks = generate_face(alpha, delta, [0, 10, 0], [0, 0, 0])
    ground_truth_landmarks = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")

    debug = 1
