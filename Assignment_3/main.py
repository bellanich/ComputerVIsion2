from pinhole_camera_model import run_pinhole_camera
from mesh_to_png import load_faces, mesh_to_png
from face_landmark_detection import face_landmark_detection

from data_def import Mesh

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from math import pi, floor, ceil

from time import time


def generate_face(alpha, delta, angles, translation):
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
    transformed_face, landmarks = run_pinhole_camera(angles, translation, pCexp, mean_tex, use_landmarks=True)

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

    diff = (landmarks - ground_truth_landmarks)**2
    landmark_loss = torch.sum(diff).float()

    alpha_regularization_loss = 0
    for i in range(0, len(alpha)):
        alpha_regularization_loss += L_alpha * alpha[i] ** 2

    delta_regularization_loss = 0
    for i in range(0, len(delta)):
        delta_regularization_loss += L_delta * delta[i] ** 2

    return landmark_loss + alpha_regularization_loss.float() + delta_regularization_loss.float()


def run_update_loop(ground_truth_landmarks, ground_truth_images):
    # This loops finds the best alpha, delta, omega and t (translation) by minimizing error in landmarks
    # init values
    alpha           = torch.tensor(np.random.uniform(-1, 1, 30), requires_grad=True)
    delta           = torch.empty(30)
    L_alpha         = 1
    L_delta         = 20
    angles          = torch.empty(3)
    translation     = torch.empty(3)
    learning_rate   = 0.01
    weight_decay    = 0.05

    # make tensors
    alpha       = nn.Parameter(alpha)
    delta       = nn.Parameter(delta)
    translation = nn.Parameter(translation)
    angles      = nn.Parameter(angles)

    # define the Adam optimizer
    optimizer = torch.optim.Adam([alpha, delta, translation, angles], lr=learning_rate, weight_decay=weight_decay)

    delta_archive = []
    translation_archive = []
    angles_archive = []

    # # go through every image in list (question 6)
    # for ii, gt_image in enumerate(list_of_gt_images):
    #     print()
    #     print('new image processing.....')
    #     # flip y values of landmarks for making them comparabel with generated landmarks - unclear why landmarks y are mirrored
    #     [_, ySize, _] = np.shape(gt_image)
    #     gt_landmarks = list_of_gt_landmarks[ii]
    #     gt_landmarks[:, 1] = ySize - gt_landmarks[:, 1]
    #     print_image(gt_image, torch.tensor(gt_landmarks), 'ground truth landmarks')

    for image_index in range(len(ground_truth_images)):

        # turn ground truth landmarks into a tensor
        ground_truth_landmarks_ = torch.tensor(ground_truth_landmarks[image_index], dtype=torch.float64)

        # initialize an archive for the loss value
        loss_archive = []

        # re-initialize delta, tranlation and rotation for each individual image
        delta = torch.tensor(np.random.uniform(-1, 1, 20), requires_grad=True)
        translation = torch.tensor([124.0, 90.0, -400.0], requires_grad=True, dtype=torch.float64)
        angles = torch.tensor([0.0, 5.0, 0.0], requires_grad=True, dtype=torch.float64)

        # loop to decrease loss
        loss_diff = 1
        old_loss = 1000000000000000000
        iteration = 0
        while abs(loss_diff) > 0.001 and iteration < 20000:

            # forward pass - calculate new values
            [landmarks, _, _, _] = generate_face(alpha, delta, angles, translation)

            # # print generated landmarks on ground truth_image
            if iteration % 500 == 0:
                print_image(ground_truth_images[image_index], landmarks, 'landmarks iteration: ' + str(iteration), 1, image_index, iteration)

            # calculate loss
            L_fit = calculate_loss(landmarks, ground_truth_landmarks_, alpha, angles, L_alpha, L_delta)

            # backward pass, L_fit
            optimizer.zero_grad()       # reset gradients
            L_fit.backward()   		    # compute the gradients

            # adjust variables
            optimizer.step()            # adjust parameters (alpha, delta)

            # update convergence parameters and display loss
            loss_diff = old_loss - L_fit
            old_loss = L_fit
            loss_archive.append(L_fit)
            if iteration % 500 == 0:
                print('iteration: {} loss: {}'.format(iteration, L_fit))

            iteration += 1

        print_image(ground_truth_images[image_index], landmarks, 'landmarks iteration: {}'.format(iteration), 1, image_index, iteration)

        delta_archive.append(delta)
        translation_archive.append(translation)
        angles_archive.append(angles)

        plt.figure()
        plt.plot(loss_archive)
        plt.savefig('./output/loss/1_{}.png'.format(image_index))
        plt.close()

    return alpha, delta_archive, angles_archive, translation_archive


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


def texture3D(gt_image, alpha, beta, omega, t, i):

    # generate face based on given parameters alpha and beta and transform this face with omage and t. 
    [_, pCexp, triangles, transformed_face] = generate_face(alpha, beta, omega, t)
    projection2D = transformed_face[:, :2]     # the 2D projected datapoints (x, y) for each of points in 3D pointcloud pCexp

    pCexp = pCexp.data.numpy()
    projection2D = projection2D.data.numpy()

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
        [_, ySize, _] = np.shape(gt_image)
        v11 = gt_image[ySize - y1, x1, :]
        v12 = gt_image[ySize - y2, x1, :]
        v21 = gt_image[ySize - y1, x2, :]
        v22 = gt_image[ySize - y2, x2, :]
        vc = calc_bilinear_interpol(x1, x2, y1, y2, xc, yc, v11, v12, v21, v22)  # vc: dim 3 vector containing RGB values
        # make this RGB value the value for pointcloud 
        projected_tex[Point, :] = vc
    # print the result - RESULTS LOOK FUNNY SINCE I HAVE CHOSEN NOT OPTIMAL PARAMETERS alpha, beta, omega, t
    mesh = Mesh(pCexp, projected_tex, triangles)
    mesh_to_png("./output/generated_faces/run_1_face_{}.png".format(i), mesh)
    return


def print_image(image, landmarks, title, run, image_index, iteration, flipy=True):
    plt.figure()
    plt.imshow(image)

    data = torch.zeros(landmarks.shape)
    data.copy_(landmarks)
    data = data.data.numpy()
    X = data[:, 0]

    if flipy:
        [_, Ysize, _] = np.shape(image)
        Y = Ysize - data[:, 1]
    else:
        Y = data[:, 1]
    plt.scatter(X, Y)	
    plt.title(title)	
    plt.savefig('./output/intermediate_images/run_{}_face_{}_{}.png'.format(run, image_index, iteration))
    plt.close()


if __name__ == "__main__":

    # get ground truth images and landmarks from images in {CWD}/Faces/ directory
    # NB: all images should be of the same person, since we will train alpha (facial identity) on these images
    list_of_gt_landmarks, list_of_gt_images = face_landmark_detection("./Data/shape_predictor_68_face_landmarks.dat", "./Faces/")

    for ii, gt_image in enumerate(list_of_gt_images):
        # flip y values of landmarks for making them comparabel with generated landmarks - unclear why landmarks y are mirrored
        [_, ySize, _] = np.shape(gt_image)
        gt_landmarks = list_of_gt_landmarks[ii]
        gt_landmarks[:, 1] = ySize - gt_landmarks[:, 1]

    # question 4
    print("Training")
    [alpha, delta_archive, angles_archive, translation_archive] = run_update_loop(list_of_gt_landmarks, list_of_gt_images)

    # question 5
    print("generating textured images")
    for i, image in enumerate(list_of_gt_images):
        texture3D(image, alpha, delta_archive[i], angles_archive[i], translation_archive[i], i)
