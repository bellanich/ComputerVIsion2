from pinhole_camera_model import run_pinhole_camera
from mesh_to_png import load_faces, mesh_to_png
from face_landmark_detection import face_landmark_detection

from data_def import Mesh

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from math import pi, floor, ceil

def generate_face(alpha, delta, omega, t):
    """
    :param alpha: a python list
    :param delta: a python list
    :param omega: a python list
    :param t: a python list
    :return:
    """
  
    # generate face given alpha and delta
    pCid, pCexp, mean_tex, triangles = load_faces(alpha, delta)

    # transform this into 2D with omega and t
    transformed_face, landmarks = run_pinhole_camera(omega, t, pCexp, mean_tex, use_landmarks=True)

    return landmarks, pCexp, triangles, transformed_face

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
    """

    diff = (torch.FloatTensor(landmarks) - torch.FloatTensor(ground_truth_landmarks))**2
    landmark_loss = torch.sum(diff).float()

    alpha_regularization_loss = 0
    for i in range(0,30):
        alpha_regularization_loss += L_alpha * alpha[i] ** 2

    delta_regularization_loss = 0
    for i in range(0,20):
        delta_regularization_loss += L_delta * delta[i] ** 2

    print('landmark_loss:', landmark_loss)

    return landmark_loss + alpha_regularization_loss.float() + delta_regularization_loss.float()

def run_update_loop():
    # This loops finds the best alpha, delta, omega and t (translation) by minimizing error in landmarks
    # init values
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    L_alpha = 5
    L_delta = 5
    transl = [180.0, 150.0, -400.0]		# according to HINT / Peters Golden Hand
    angles = [0.0, 0.0, 180.0]

    # get ground truth and print it
    ground_truth_landmarks, ground_truth_image = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")
    print_image(ground_truth_image, ground_truth_landmarks, 'ground truth landmarks')

    # make tensors
    alpha = torch.tensor(alpha, requires_grad=True)    
    delta = torch.tensor(delta, requires_grad=True)
    transl = torch.tensor(transl, requires_grad=True)
    angles = torch.tensor(angles, requires_grad=True)

    # loop to decrease loss
    for loop in range(3):
    # TODO - the overall code is ok - build convergence criterium 
    # TODO - now the loss does not contain landmark_loss. Landmark_loss seems way to large and not converging - so this needs to debugging. -  
    # DONE - ensure the correct initialisation - AND: see HINT (translation by -400)

        # .detach to get rid of the gradients, needed since 'generate_face only eats np.arrays (A, B, O, t)
        A = alpha.detach().numpy()   
        B = delta.detach().numpy()
        O = angles.detach().numpy()
        t = transl.detach().numpy()       

        # forward pass - calculate new values
        [landmarks, _, _, _] = generate_face(A, B, O, t)

        # print generated landmarks on ground truth_image
        if loop % 20 == 0:
            print_image(ground_truth_image, landmarks, 'landmarks iteration: ' + str(loop))

        # define the Adam optimizer
        optimizer = torch.optim.Adam([alpha, delta, transl, angles], lr=0.5)

	# calculate loss
        L_fit = calculate_loss(landmarks, ground_truth_landmarks, alpha, delta, L_alpha, L_delta)
        print('loss:', L_fit)

        # backward pass
        optimizer.zero_grad()           # reset gradients
        L_fit.backward()   		# compute the gradients 

        # adjust variables
        optimizer.step()                # adjust parameters (alpha, delta)

    return A, B, O, t

def calc_bilinear_interpol(x1, x2, y1, y2, xc, yc, v11, v12, v21, v22):
    # function is build for question 5 - NOT YET USED
    '''
    calculates the some value vc on coordinates [xc, yc], based on the 4 values on the surrounding corners x1, x2, y1, y2.
    v11, v12, v21, v22 any float
    '''
    # step 1 - check that cx is in between the surrounding corner coordinates
    if (xc > x1 and x2 >= xc and yc > y1 and y2 >= yc):
        # step 2 - calc the bilinear interpolation
        vc1 = (x2-xc)/(x2-x1) * v11 + (xc-x1)/(x2-x1) * v21
        vc2 = (x2-xc)/(x2-x1) * v12 + (xc-x1)/(x2-x1) * v22
        vc =  (y2-yc)/(y2-y1) * vc1 + (yc-y1)/(y2-y1) * vc2
        return vc
    else:
        print('Error: wrong coordinates')
        return
  
def texture3D(alpha, beta, omega, t):
    # get ground truth image (again): 784 (x) X 1024 (y) x 3 (RGB)
    _, gt_image = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")

    # generate face based on given parameters alpha and beta and transform this face with omage and t. 
    [landmarks, pCexp, triangles, transformed_face] = generate_face(alpha, beta, omega, t)
    projection2D = transformed_face[:, :2]     # the 2D projected datapoints (x, y) for each of points in 3D pointcloud pCexp

    # create texture file:
    [noPoints, _] = np.shape(projection2D)
    projected_tex = np.zeros((noPoints, 3))   # same size as mean_tex

    # for each point in 2Dprojected generated face, find its texture from the original gt_image by bilinear interpolation
    for Point in range(noPoints):
        [xc, yc] = projection2D[Point, :]
        x1 = floor(xc)
        x2 = ceil(xc)
        y1 = floor(yc)
        y2 = ceil(yc)
        v11 = gt_image[y1, x1, :]
        v12 = gt_image[y2, x1, :]
        v21 = gt_image[y1, x2, :]
        v22 = gt_image[y2, x2, :]
        vc = calc_bilinear_interpol(x1, x2, y1, y2, xc, yc, v11, v12, v21, v22)  # vc: dim 3 vector containing RGB values
        # make this RGB value the value for pointcloud 
        projected_tex[Point, :] = vc
    # print the result - RESULTS LOOK FUNNY SINCE I HAVE CHOSEN NOT OPTIMAL PARAMETERS alpha, beta, omega, t
    mesh = Mesh(pCexp, projected_tex, triangles)
    mesh_to_png("pCexp_P.png", mesh)
    return

def print_image(image, landmarks, title):
    plt.figure()
    plt.imshow(image)
    X = landmarks[:, 0]
    Y = landmarks[:, 1]
    plt.scatter(X, Y)	
    plt.title(title)	
    plt.show()    

if __name__ == "__main__":

    # question 4\
    [A, B, O, t] = run_update_loop()

    # question 5
    texture3D(A, B, O, t)



