import numpy as np

def transform_vertices(mat, vertices):
    homo_vertices = np.pad(np.array(vertices), ((0, 0), (0, 1)), mode='constant', constant_values=1)
    trans_vertices = (np.matmul(mat, homo_vertices.T).T)[:, :3]
    return trans_vertices

def transform_batch_vertices(mat, batch_vertices):
    homo_vertices = np.pad(np.array(batch_vertices), ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
    trans_vertices = (np.matmul(mat, homo_vertices.T).T)[:, :3]
    return trans_vertices