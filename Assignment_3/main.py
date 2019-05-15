from pinhole_camera_model import run_pinhole_camera
from mesh_to_png import load_faces
from face_landmark_detection import face_landmark_detection

import numpy as np
import torch
import torch.nn as nn

def generate_face(alpha, delta, omega, t):
    """
    :param alpha: a python list
    :param delta: a python list
    :param omega: a python list
    :param t: a python list
    :return:
    """
    pCid, pCexp, mean_tex, triangles = load_faces(alpha, delta)
    print('load_faces done.')
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
    TODO: expand the loss function to compare 2D vectors in stead of scalars: DONE
    """

    diff = (torch.tensor(landmarks).float() - torch.tensor(ground_truth_landmarks).float())**2
    landmark_loss = torch.sum(torch.tensor(diff))       

    alpha_regularization_loss = 0
    for i in range(0,30):
        alpha_regularization_loss += L_alpha * alpha[i] ** 2

    delta_regularization_loss = 0
    for i in range(0,20):
        delta_regularization_loss += L_delta * delta[i] ** 2

    return landmark_loss + torch.tensor(alpha_regularization_loss).float() + torch.tensor(delta_regularization_loss).float()


def run_update_loop():
    """Use tensorflow as described in the assignment
    TODO: convert ground_truth_landmarks into XY-coordinates - This is now done in face_landmark_detection.py
    """
    ground_truth_landmarks = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")
    pass


if __name__ == "__main__":

    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    L_alpha = 0.5
    L_delta = 0.5

    ground_truth_landmarks = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")

    for loop in range(5):
    # Start loop for improving Loss - this loops works but does not decrease L_fit yet .
    # TODO - build convergence criterium

        # I believe we need to have alpha and delta as tensors with gradients from the beginning
        # and ensure that the function 'generate_face' is robust for tensors 
        # is this true ? lots of work). 

        # forward pass - calculate new values
        landmarks = generate_face(alpha, delta, [0, 10, 0], [0, 0, 0])

        # define the Adam optimizer
        alpha = torch.tensor(alpha, requires_grad=True)    
        delta = torch.tensor(delta, requires_grad=True)	
        optimizer = torch.optim.Adam([alpha, delta], lr=0.5)

	# calculate loss
        # landmarks = torch.tensor(landmarks, requires_grad=True)
        L_fit = calculate_loss(landmarks, ground_truth_landmarks, alpha, delta, L_alpha, L_delta)
        # L_fit = torch.autograd.Variable(L_fit, requires_grad=True)      # do not need this anymore, since function 'calculate loss' delivers tensor L_fit
        print('loss:', L_fit)

        # backward pass
        optimizer.zero_grad()           # reset gradients
        L_fit.backward()   		# compute the gradients 

        # adjust variables
        optimizer.step()                # adjust parameters (alpha, delta)

        # to make it np arrays again, needed for forward pass - function 'generate_face'
        alpha = alpha.detach().numpy()   
        delta = delta.detach().numpy()


