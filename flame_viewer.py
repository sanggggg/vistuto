import faulthandler

from src.utils.meshes import get_sunglass_meshes
faulthandler.enable()

import src.flamelib.flame_runner as flame_runner
import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import threading
import numpy as np
import time

class FlameWindow:
    is_done = False
    runner = flame_runner.FlameRunner()
    shape_params = torch.zeros(1, 100).cuda()
    pose_params = torch.zeros(1, 6, dtype=torch.float32).cuda()
    neck_pose_params = torch.zeros(1, 3, dtype=torch.float32).cuda()
    expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
    sunglasses_params = np.array([0, 0.027, 0.058])
    is_updating = False

    def __init__(self):

        # Setup window
        self.window = gui.Application.instance.create_window(
            "FlameViewer", 2000, 1000)
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
        self._panel.add_child(gui.Label("Shape Params"))
        for i in range(10):
            slider = gui.Slider(gui.Slider.Type.DOUBLE)
            slider.set_limits(-3.0, 3.0)
            self._shape_params.append(slider)
            slider.set_on_value_changed(self.param_change_callback('shape', i))
            self._panel.add_child(slider)
        self._panel.add_fixed(0.5 * em)
        
        self._pose_params = []
        self._panel.add_child(gui.Label("Pose Params"))
        for i in range(6):
            slider = gui.Slider(gui.Slider.Type.DOUBLE)
            slider.set_limits(-3.0, 3.0)
            self._pose_params.append(slider)
            slider.set_on_value_changed(self.param_change_callback('pose', i))
            self._panel.add_child(slider)
        self._panel.add_fixed(0.5 * em)

        self._neck_pose_params = []
        self._panel.add_child(gui.Label("Neck Pose Params"))
        for i in range(3):
            slider = gui.Slider(gui.Slider.Type.DOUBLE)
            slider.set_limits(-3.0, 3.0)
            self._neck_pose_params.append(slider)
            slider.set_on_value_changed(self.param_change_callback('neck_pose', i))
            self._panel.add_child(slider)
        self._panel.add_fixed(0.5 * em)


        self._expression_params = []
        self._panel.add_child(gui.Label("Expression Params"))
        for i in range(10):
            slider = gui.Slider(gui.Slider.Type.DOUBLE)
            slider.set_limits(-3.0, 3.0)
            self._expression_params.append(slider)
            slider.set_on_value_changed(self.param_change_callback('expression', i))
            self._panel.add_child(slider)
        self._panel.add_fixed(0.5 * em)

        self._show_sunglasses = gui.Checkbox("Show Sunglasses")
        self._show_sunglasses.set_on_checked(self.show_sunglasses)
        self._panel.add_child(self._show_sunglasses)
        self._panel.add_fixed(0.5 * em)
        self.should_show_sunglasses = False

        self._sunglass_params = []
        self._panel.add_child(gui.Label("Sunglass Params"))
        for i in range(3):
            slider = gui.Slider(gui.Slider.Type.DOUBLE)
            slider.set_limits(-0.1, 0.1)
            slider.double_value = self.sunglasses_params[i]
            self._sunglass_params.append(slider)
            slider.set_on_value_changed(self.param_change_callback('sunglasses', i))
            self._panel.add_child(slider)
        self._panel.add_fixed(0.5 * em)
        self._panel.add_fixed(0.5 * em)

        self.window.add_child(self._panel)

        # Mat
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.calculate(reset_camera=True)
    
    def param_change_callback(self, typ, idx):
        def inner(value):
            if typ == 'shape':
                self.shape_params[0, idx] = value
            elif typ == 'pose':
                self.pose_params[0, idx] = value
            elif typ == 'neck_pose':
                self.neck_pose_params[0, idx] = value
            elif typ == 'expression':
                self.expression_params[0, idx] = value
            elif typ == 'sunglasses':
                self.sunglasses_params[idx] = value
            
            threading.Thread(target=self.calculate).start()
        return inner

    def on_layout(self, layout_context):
        contentRect = self.window.content_rect
        panel_width = 15 * layout_context.theme.font_size  # 15 ems wide
        self._scene.frame = gui.Rect(contentRect.x, contentRect.y,
                                       contentRect.width - panel_width,
                                       contentRect.height)
        self._panel.frame = gui.Rect(self._scene.frame.get_right(),
                                    contentRect.y, panel_width,
                                    contentRect.height)
    
    def on_close(self):
        self.is_done = True
        return True
    
    def show_sunglasses(self, value):
        self.should_show_sunglasses = value
        threading.Thread(target=self.calculate).start()
    
    def calculate(self, reset_camera=False):
        if self.is_updating:
            return
        self.is_updating = True
        vertices, faces, trans = self.runner.run(self.shape_params, self.pose_params, self.neck_pose_params, self.expression_params)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        mesh.compute_vertex_normals()
        sunglasses = get_sunglass_meshes(self.sunglasses_params)
        sunglasses = transform_mesh(trans[0][1].detach().cpu().numpy(), sunglasses)
        sunglasses.compute_vertex_normals()

        def update():
            self._scene.scene.clear_geometry()
            self._scene.scene.add_geometry("flame", mesh, self.mat)
            if self.should_show_sunglasses:
                self._scene.scene.add_geometry("suglasses", sunglasses, self.mat)
            if reset_camera:
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60.0, bounds, bounds.get_center())
            self.is_updating = False
        gui.Application.instance.post_to_main_thread(self.window, update)

def transform_mesh(mat, mesh):
    homo_vertices = np.pad(np.array(mesh.vertices), ((0, 0), (0, 1)), mode='constant', constant_values=1)
    trans_vertices = (np.matmul(mat, homo_vertices.T).T)[:, :3]
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(trans_vertices),
        mesh.triangles,
    )

if __name__ == "__main__":
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = FlameWindow()
    app.run()
