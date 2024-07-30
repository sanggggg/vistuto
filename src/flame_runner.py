import torch

from .FLAME import FLAME

class Cfg:
    def __init__(self):
        self.flame_model_path = 'models/male_model.pkl'  # acquire it from FLAME project page
        self.static_landmark_embedding_path = 'models/flame_static_embedding.pkl'  # acquire it from FLAME project page
        self.dynamic_landmark_embedding_path = 'models/flame_dynamic_embedding.npy'  # acquire it from RingNet project page
        self.shape_params = 100
        self.expression_params = 50
        self.pose_params = 6
        self.use_face_contour = True
        self.use_3D_translation = True
        self.optimize_eyeballpose = False
        self.optimize_neckpose = False
        self.batch_size = 1

class FlameRunner:
    def __init__(self):
        config = Cfg()
        self.flamelayer = FLAME(config)
        self.flamelayer.cuda()
    
    def run(self, shape_params, pose_params, expression_params):
        neck_pose = torch.zeros(1, 3).cuda()
        eye_pose = torch.zeros(1, 6).cuda()
        vertice, landmark = self.flamelayer(
            shape_params, expression_params, pose_params, neck_pose, eye_pose
        )
        faces = self.flamelayer.faces
        vertices = vertice[0].detach().cpu().numpy().squeeze()
        return vertices, faces