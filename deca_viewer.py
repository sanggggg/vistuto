from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg
from src.decalib.datasets import datasets
import torch
import open3d as o3d
import argparse

def main(args):
    testdata = datasets.TestData(args.inputpath, iscrop=True, crop_size=224, face_detector='fan', sample_step=10)
    deca = DECA(config=deca_cfg, device='cuda')
    for data in testdata:
        name = data['imagename']
        images = data['image'].to('cuda')[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict)
        deca.save_obj("asdf.obj", opdict)
        vertices = opdict['verts'][0].cpu().numpy()
        faces = deca.render.faces[0].cpu().numpy()
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')        
    main(parser.parse_args())