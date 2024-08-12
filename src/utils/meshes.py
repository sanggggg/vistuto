import open3d as o3d
import numpy as np
from .matrix_transform import transform_mesh

def get_sunglass_meshes(params = [0, 0.027, 0.058]):
    sunglasses = transform_mesh(np.array(
        [[0.1, 0, 0, params[0]],
            [0, 0.1, 0, params[1]],
            [0, 0, 0.1, params[2]],
            [0, 0, 0, 1],
        ]
    ), o3d.io.read_triangle_model("assets/Sunglasses.obj").meshes[0].mesh)
    return sunglasses