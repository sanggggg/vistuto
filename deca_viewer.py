from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg
from src.decalib.datasets import datasets
import numpy as np
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import argparse


class DECAWindow:
    def __init__(self, timeline, faces):
        self.timeline = timeline
        self.faces = faces
        # Setup window
        self.window = gui.Application.instance.create_window(
            "DECAViewer", 2000, 1000)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(self.on_close)
        em = self.window.theme.font_size
        margin = 0.5 * em

        # Setup 3D Scene
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.show_axes(True)
        self.window.add_child(self._scene)

        # Setup panel
        self._panel = gui.Vert(0.5 * em, gui.Margins(margin))

        self._shape_params = []
        self._panel.add_child(gui.Label("Video Frame"))
        frame_slider = gui.Slider(gui.Slider.Type.INT)
        frame_slider.set_limits(0, len(timeline)-1)
        frame_slider.set_on_value_changed(self.frame_change)
        self._panel.add_child(frame_slider)
        self._panel.add_fixed(0.5 * em)

        # Setup Origin Image Panel
        self._origin_image = gui.ImageWidget(self.timeline[0]['original'])
        self._panel.add_child(self._origin_image)

        # Setup Rendered Image Panel
        self._origin_rendered_image = gui.ImageWidget(self.timeline[0]['rendered_on_original'])
        self._panel.add_child(self._origin_rendered_image)

        self.window.add_child(self._panel)

        # Mat
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.frame_change(0, reset_camera=True)
    
    def on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size  # 15 ems wide
        self._scene.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width,
                                       contentRect.height)
        self._panel.frame = gui.Rect(self._scene.frame.get_right(),
                                    contentRect.y, panel_width,
                                    contentRect.height)
        
    def frame_change(self, value, reset_camera=False):
        vertices = self.timeline[int(value)]['vertices']
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(self.faces),
        )
        mesh.compute_vertex_normals()
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry("deca", mesh, self.mat)
        self._origin_image.update_image(self.timeline[int(value)]['original'])
        self._origin_rendered_image.update_image(self.timeline[int(value)]['rendered_on_original'])
        if reset_camera:
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60.0, bounds, bounds.get_center())
    
    def on_close(self):
        self.is_done = True
        return True


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[0,1,2]]
    return image.astype(np.uint8).copy()

def main(args):
    testdata = datasets.TestData(args.inputpath, iscrop=True, crop_size=224, face_detector='fan', sample_step=5)
    deca_cfg.model.is_cool = args.cool
    deca = DECA(config=deca_cfg, device='cuda')

    total_frames = len(testdata)
    batch_size = 5
    timeline = []

    for i in range((total_frames - 1) // batch_size + 1):
        if i == total_frames // batch_size:
            batch_data = [testdata[x] for x in range(i*batch_size, total_frames)]
        else:
            batch_data = [testdata[x] for x in range(i*batch_size, (i+1)*batch_size)]
        images = torch.stack([data['image'] for data in batch_data]).to('cuda')
        original_images = torch.stack([data['original_image'] for data in batch_data]).to('cuda')
        tforms = torch.stack([torch.inverse(data['tform']).transpose(0,1) for data in batch_data]).to('cuda')

        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict, render_orig=True, tform=tforms, original_image=original_images)
        vertices = opdict['verts'].cpu().numpy()
        faces = deca.render.faces[0].cpu().numpy()
        for j in range(len(images)):
            timeline.append(
                {
                    'vertices': vertices[j],
                    'original': o3d.geometry.Image(tensor2image(images[j])),
                    'rendered_on_original': o3d.geometry.Image(tensor2image(visdict['shape_images'][j])),
                }
            )
    
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = DECAWindow(timeline, faces)
    app.run()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')        
    parser.add_argument('--cool', action='store_true', help='cool mode')
    main(parser.parse_args())