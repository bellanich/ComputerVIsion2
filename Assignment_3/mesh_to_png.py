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

newm = mean_shape + newcombi
print(np.shape(newm))

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
    # mesh = Mesh(mean_shape, mean_tex, triangles)
    mesh = Mesh(newm, mean_tex, triangles)
    mesh_to_png("output.png", mesh)
