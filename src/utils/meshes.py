import open3d as o3d
import numpy as np
from .matrix_transform import transform_vertices

def get_sunglass_meshes(params = [0, 0.027, 0.058]):
    sunglass_mesh = o3d.io.read_triangle_model("assets/Sunglasses.obj").meshes[0].mesh
    trans_vertices = transform_vertices(np.array(
        [[0.1, 0, 0, params[0]],
            [0, 0.1, 0, params[1]],
            [0, 0, 0.1, params[2]],
            [0, 0, 0, 1],
        ]
    ), sunglass_mesh.vertices)
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(trans_vertices),
        sunglass_mesh.triangles,
    )