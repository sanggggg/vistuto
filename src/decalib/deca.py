# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread

from ..utils.meshes import get_sunglass_meshes
from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from ..flamelib.FLAME import FLAME
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .utils.config import cfg
torch.backends.cudnn.benchmark = True

class DECA(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(DECA, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size, rasterizer_type=self.cfg.rasterizer_type, is_cool=model_cfg.is_cool).to(self.device)

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}

        self.sunglasses_vertices = np.array(get_sunglass_meshes().vertices)

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, out_scale=model_cfg.max_z, sample_mode = 'bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check model path: {model_path}')
            # exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    # @torch.no_grad()
    def encode(self, images):
        parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:,3:].clone() # x for yaw (open mouth), y for pitch (left ang right), z for roll
            posecode[:,3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose  
        return codedict

    # @torch.no_grad()
    def decode(self, codedict, rendering=True, iddict=None, return_vis=True, use_detail=True,
                render_orig=False, original_image=None, tform=None):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, rotation = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])

        if self.cfg.model.is_cool:
            rotation = rotation[:,1]
            glasses_verts = self.render.sunglasses_vertices.to(self.device).repeat(batch_size, 1, 1)
            glasses_verts = torch.cat((glasses_verts, torch.ones_like(glasses_verts[:, :, :1])), dim=-1)
            glasses_verts = rotation @ torch.Tensor([
                [0.09, 0, 0, 0.0],
                [0, 0.09, 0, 0.027],
                [0, 0, 0.09, 0.058],
                [0, 0, 0, 1],
            ]).to(self.device) @ glasses_verts.transpose(1,2)
            glasses_verts = glasses_verts.transpose(1,2)[:,:,:3]
            verts = torch.cat((verts, glasses_verts), dim=1)

        albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 

        ## projection
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
        }

        ## rendering
        if return_vis and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            # import ipdb; ipdb.set_trace()
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            background = original_image
            images = original_image
        else:
            h, w = self.image_size, self.image_size
            background = None

        if rendering:
            # ops = self.render(verts, trans_verts, albedo, codedict['light'])
            ops = self.render(verts, trans_verts, albedo, h=h, w=w, background=background)
            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']
        
        if return_vis:
            ## render shape
            shape_images, _, grid, alpha_images = self.render.render_shape(verts, trans_verts, h=h, w=w, images=background, return_grid=True)
            visdict = {
                'inputs': images, 
                'shape_images': shape_images,
            }

            return opdict, visdict

        else:
            return opdict
