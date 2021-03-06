#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy

# COMMENT FROM Peter
# RUN THIS CODE FROM COMMAND WITH:
# >> python3 face_landmark_detection.py ./Data/shape_predictor_68_face_landmarks.dat .
# First argument is full path to .dat file, second argument is path to image.


import sys
import os
import dlib
import glob
import numpy as np
import pickle

# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor from:\n"
#         "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()
#
# predictor_path = sys.argv[1]
# faces_folder_path = sys.argv[2]


# From canvas
def shape_to_np(shape, dtype="int"):

    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
    return coords


def face_landmark_detection(predictor_path, faces_folder_path):
    # return a list of images and a list of landmarks

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    win = dlib.image_window()

    list_of_images = []
    list_of_landmarks = []

    for f in glob.glob(os.path.join(faces_folder_path, "*.png")):            # specify image
        print("Processing file: {}".format(f))
        img = dlib.load_rgb_image(f)

        win.clear_overlay()
        win.set_image(img)        # displays image

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                      shape.part(1)))
            # Draw the face landmarks on the screen.
            win.add_overlay(shape)

        win.add_overlay(dets)
        dlib.hit_enter_to_continue()

        landmarks = shape_to_np(shape)
        list_of_images.append(img)
        list_of_landmarks.append(landmarks)

    print('number of images loaded: ', len(list_of_images))

    with open('landmarks_bella_smile.p', 'wb') as f:
        pickle.dump(list_of_landmarks, f)


    return list_of_landmarks, list_of_images


if __name__ == "__main__":

    predictor_path = "./Data/shape_predictor_68_face_landmarks.dat"
    faces_folder_path = "./Faces/"

    face_landmark_detection(predictor_path, faces_folder_path)
