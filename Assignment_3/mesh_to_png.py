import os
import numpy as np
import trimesh
import pyrender
import h5py
from data_def import PCAModel, Mesh


def load_faces(alpha, delta):

    bfm = h5py.File("./Data/model2017-1_face12_nomouth.h5", 'r')

    # here we load shape/model/mean and convert 3N --> N x 3

    mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
    size_mean_shape = np.shape(mean_shape)

    # here we load the pcabasis and pcavar (reshape it into right tensors, take squareroot of sigma, matrix multiply it.

    pcaB_shape = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    nrDim = np.shape(pcaB_shape)[1]
    pcaB_shape = pcaB_shape.reshape(-1, 3, nrDim)
    pcaB_shape = pcaB_shape[:, :, :30]

    pcaVar_shape = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
    pcaVar_shape = pcaVar_shape[:30]

    sigma_shape = alpha * np.sqrt(pcaVar_shape)
    sigma_shape = np.sqrt(pcaVar_shape)

    E_shape_sigma = np.matmul(pcaB_shape, sigma_shape)

    # here we load expression/model/mean and convert 3N --> N x 3

    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3))

    # here we load for expression: the pcabasis and pcavar (reshape it into right tensors, take squareroot of sigma, matrix multiply it.

    pcaB_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    nrDim = np.shape(pcaB_exp)[1]
    pcaB_exp = pcaB_exp.reshape(-1, 3, nrDim)

    pcaVar_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)

    pcaB_exp = pcaB_exp[:, :, :20]
    pcaVar_exp = pcaVar_exp[:20]

    sigma_exp = delta * np.sqrt(pcaVar_exp)
    E_exp_sigma = np.matmul(pcaB_exp, sigma_exp)

    # combine mean_shape with Eid*[alpha*sigma]
    pCid = mean_shape + E_shape_sigma
    pCexp = mean_shape + E_shape_sigma + mean_exp + E_exp_sigma

    # this is standard code to read color/tex and triangles
    mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
    triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

    return pCid, pCexp, mean_tex, triangles


def mesh_to_png(file_name, mesh):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)
    return


if __name__ == '__main__':
    # output: only mean_shape)
    # mesh = Mesh(mean_shape, mean_tex, triangles)

    alpha = np.random.uniform(-1, 1, (30))
    delta = np.random.uniform(-1, 1, (20))
    [pCidentity, pCexpression, mean_tex, triangles] = load_faces(alpha, delta)

    # output: mean_shape + Eid*[alpha*sigma]
    mesh = Mesh(pCidentity, mean_tex, triangles)
    # mesh_to_png("pCid.png", mesh)

    # output: all combined
    mesh = Mesh(pCexpression, mean_tex, triangles)
    mesh_to_png("pCexp.png", mesh)

