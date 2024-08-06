import torch

from .FLAME import FLAME

class Cfg:
    def __init__(self):
        self.flame_model_path = 'data/male_model.pkl'  # acquire it from FLAME project page
        self.shape_params = 100
        self.expression_params = 50
        self.pose_params = 6
        self.use_3D_translation = True
        self.batch_size = 1

class FlameRunner:
    def __init__(self):
        config = Cfg()
        self.flamelayer = FLAME(config)
        self.flamelayer.cuda()
    
    def run(self, shape_params, pose_params, neck_pose_params, expression_params):
        eye_pose = torch.zeros(1, 6).cuda()
        vertice, trans = self.flamelayer(
            shape_params, expression_params, pose_params, neck_pose_params, eye_pose
        )
        faces = self.flamelayer.faces
        vertices = vertice[0].detach().cpu().numpy().squeeze()
        return vertices, faces, trans