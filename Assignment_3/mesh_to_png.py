import os
import numpy as np
import trimesh
import pyrender

import h5py

from data_def import PCAModel, Mesh

# Peter: added .
bfm = h5py.File("./Data/model2017-1_face12_nomouth.h5", 'r')

# here we load shape/model/mean and convert 3N --> N x 3

mean_shape = np.asarray(bfm['shape/model/mean'], dtype=np.float32).reshape((-1, 3))
size_mean_shape = np.shape(mean_shape)
print(size_mean_shape)

# here we load the pcabasis and pcavar (reshape it into right tensors, take squareroot of sigma, matrix multiply it. 

pcaB_shape = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
nrDim = np.shape(pcaB_shape)[1]
pcaB_shape = pcaB_shape.reshape(-1, 3, nrDim)
size_pcaB_shape = np.shape(pcaB_shape)
print(size_pcaB_shape)

pcaVar_shape = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
size_pcaVar_shape = np.shape(pcaVar_shape)
print(size_pcaVar_shape)

newshape = pcaB_shape[:, :, :30]
newVar = pcaVar_shape[:30]
print(np.shape(newshape))
print(np.shape(newVar))

newcombi = np.matmul(newshape, np.sqrt(newVar))
print(np.shape(newcombi))

# here we load expression/model/mean and convert 3N --> N x 3

mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32).reshape((-1, 3))

# here we load for expression: the pcabasis and pcavar (reshape it into right tensors, take squareroot of sigma, matrix multiply it. 

pcaB_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
nrDim = np.shape(pcaB_exp)[1]
pcaB_exp = pcaB_exp.reshape(-1, 3, nrDim)
print('pcaB_exp: ', np.shape(pcaB_exp))

pcaVar_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)

size_pcaVar_shape = np.shape(pcaVar_shape)
print('pcaVar_exp: ', np.shape(pcaVar_exp))

pcaB_exp = pcaB_exp[:, :, :20]
pcaVar_exp = pcaVar_exp[:20]
delta = zeros(20)
E_exp_sigma = np.matmul(pcaB_exp, np.sqrt(pcaVar_exp))
print('E_exp_sigma: ', np.shape(E_exp_sigma))

# combine mean_shape with Eid*[alpha*sigma] -TODO nog laatste twee toevoegen
pointCloud2 = mean_shape + newcombi
pointCloud3 = mean_shape + newcombi + mean_exp + E_exp_sigma

# this is standard code to read color/tex and triangles
mean_tex = np.asarray(bfm['color/model/mean'], dtype=np.float32).reshape((-1, 3))
triangles = np.asarray(bfm['shape/representer/cells'], dtype=np.int32).T

def mesh_to_png(file_name, mesh):
    mesh = trimesh.base.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.triangles,
        vertex_colors=mesh.colors)

    png = mesh.scene().save_image()
    with open(file_name, 'wb') as f:
        f.write(png)

if __name__ == '__main__':
    # output: only mean_shape)
    # mesh = Mesh(mean_shape, mean_tex, triangles)

    # output: mean_shape + Eid*[alpha*sigma]
    # mesh = Mesh(pointCloud2, mean_tex, triangles)
    # mesh_to_png("pointCloud2.png", mesh)

    # output: all combined
    mesh = Mesh(pointCloud3, mean_tex, triangles)
    mesh_to_png("pointCloud3.png", mesh)
