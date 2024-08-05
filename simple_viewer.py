import numpy as np
import argparse
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import threading
import time

class App:
    def __init__(self, width, height):
        self.window = gui.Application.instance.create_window(
            title="SimpleViewer",
            width=width,
            height=height,
            x=0,
            y=0,
        )
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.set_background([255,255,255,1.0])
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
        self.record_interval = 0.05

        em = self.window.theme.font_size
        self.separation_height = int(round(0.5 * em))
        separation_height = self.separation_height
        self._settings_panel = gui.Vert(0, gui.Margins(0.25*em, 0.25*em, 0.5*em, 0.25*em))

        # View controls
        view_ctrls = gui.CollapsableVert("View Controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._view_ctrls = view_ctrls

        self._show_axis = gui.Checkbox("Show Axis")
        self._show_axis.set_on_checked(self.on_show_axis)
        self._show_grid_plane = gui.Checkbox("Show Grid Plain")
        self._show_grid_plane.set_on_checked(self.on_show_grid_plane)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_axis)
        view_ctrls.add_fixed(separation_height)
        view_ctrls.add_child(self._show_grid_plane)

        # Geometry controls
        geometry_ctrls = gui.CollapsableVert("Geometry Controls", 0.25 * em, gui.Margins(em, 0, 0, 0))
        self._geometry_ctrls = geometry_ctrls

        # Save camera trajectory
        self._save_camera = gui.Button("Record Camera Trajectory")
        self._save_camera.set_on_clicked(self.toggle_record_trajaectory)
        self._settings_panel.add_child(self._save_camera)

        # Add plane
        self._settings_panel.add_child(view_ctrls)
        self._settings_panel.add_fixed(separation_height * 2)
        self._settings_panel.add_child(geometry_ctrls)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)
        self.window.set_on_layout(self._on_layout)

        if gui.Application.instance.menubar is None:
            MENU_OPEN = 1
            MENU_TRAJ_OPEN = 2
            file_menu = gui.Menu()
            file_menu.add_item("Open Model", MENU_OPEN)
            file_menu.add_item("Open Trajectory", MENU_TRAJ_OPEN)
            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            gui.Application.instance.menubar = menu
        self.window.set_on_menu_item_activated(MENU_OPEN, self.on_menu_open)
        self.window.set_on_menu_item_activated(MENU_TRAJ_OPEN, self.on_menu_traj_open)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 12 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                              height)
    
    def on_show_axis(self, checked):
        self._scene.scene.show_axes(checked)

    def on_show_grid_plane(self, checked):
        if checked:
            self.add_plane()
        else:
            self._scene.scene.remove_geometry("plane")
    
    def on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(".obj", "Triangle mesh files (.obj)")
        dlg.add_filter("", "All files")
        def on_done(filename):
            self.window.close_dialog()
            self.load(filename)
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(on_done)
        self.window.show_dialog(dlg)
    
    def on_menu_traj_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(".npy", "Numpy files (.obj)")
        def on_done(filename):
            self.window.close_dialog()
            traj = np.load(filename)
            self.replay_camera_trajectory(traj)
        dlg.set_on_cancel(lambda: self.window.close_dialog())
        dlg.set_on_done(on_done)
        self.window.show_dialog(dlg)

    def add_plane(self, resolution=128, bound=100, up_vec='y'):
        def makeGridPlane(bound=100., resolution=128, color = np.array([0.5,0.5,0.5]), up='z'):
            min_bound = np.array([-bound, -bound])
            max_bound = np.array([bound, bound])
            xy_range = np.linspace(min_bound, max_bound, num=resolution)
            grid_points = np.stack(np.meshgrid(*xy_range.T), axis=-1).astype(np.float32) # asd
            if up == 'z':
                grid3d = np.concatenate([grid_points, np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1)], axis=2)
            elif up == 'y':
                grid3d = np.concatenate([grid_points[:,:,0][:,:,None], np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points[:,:,1][:,:,None]], axis=2)
            elif up == 'x':
                grid3d = np.concatenate([np.zeros_like(grid_points[:,:,0]).reshape(resolution, resolution, 1), grid_points], axis=2)
            else:
                print("Up vector not specified")
                return None
            grid3d = grid3d.reshape((resolution**2,3))
            indices = []
            for y in range(resolution):
                for x in range(resolution):  
                    corner_idx = resolution*y + x 
                    if x + 1 < resolution:
                        indices.append((corner_idx, corner_idx + 1))
                    if y + 1 < resolution:
                        indices.append((corner_idx, corner_idx + resolution))

            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(grid3d),
                lines=o3d.utility.Vector2iVector(indices),
            )
            line_set.paint_uniform_color(color)
            
            return line_set
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        plane = makeGridPlane(bound, resolution, up=up_vec)
        self._scene.scene.add_geometry("plane", plane, mat)
        return
    
    geom_names = []

    def get_camera_extrinsic_matrix(self):
        vm = self._scene.scene.camera.get_view_matrix()
        gl = np.array([
            [1.0, 0, 0, 0,],
            [0, -1.0, 0, 0,],
            [0, 0, -1.0, 0,],
            [0, 0, 0, 1.0,]
        ])
        return gl @ vm
    
    camera_trajectory = []
    isRecordingTrajectory = False

    def toggle_record_trajaectory(self):
        if self.isRecordingTrajectory:
            self._save_camera.text = "Record Camera Trajectory"
            self.isRecordingTrajectory = False
            self.save_dialog_open(np.copy(np.array(self.camera_trajectory)))
            self.camera_trajectory = []
        else:
            self._save_camera.text = "Save Camera Trajectory"
            self.isRecordingTrajectory = True
            threading.Thread(target=self._update_thread).start()
    
    def _update_thread(self):
        def record_camera_matrix():
            im = self.intrinsic.intrinsic_matrix
            im = np.pad(im, ((0, 1), (0, 1)), mode='constant', constant_values=0)
            em = self.get_camera_extrinsic_matrix()
            self.camera_trajectory.append((im, em))
        while self.isRecordingTrajectory:
            if self.isRecordingTrajectory:
                gui.Application.instance.post_to_main_thread(self.window, record_camera_matrix)
            time.sleep(self.record_interval)
    
    def replay_camera_trajectory(self, trajectory):
        threading.Thread(target=self._update_camera, args=[trajectory]).start()
    
    def _update_camera(self, trajectory):
        # This is NOT the UI thread, need to call post_to_main_thread() to update
        # the scene or any part of the UI.
        idx = 0
        bounds = self._scene.scene.bounding_box

        while idx < len(trajectory):
            im = trajectory[idx][0]
            im = im[:3, :3]
            em = trajectory[idx][1]
            def play_camera_trajectory():
                self._scene.setup_camera(im, em, 640, 480, bounds)
            gui.Application.instance.post_to_main_thread(self.window, play_camera_trajectory)
            idx += 1
            time.sleep(self.record_interval)
    
    def save_dialog_open(self, trajectory):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".npy", "Numpy files (.npy)")
        dlg.set_on_cancel(lambda : self.window.close_dialog())
        def save_file(path):
            self.window.close_dialog()
            np.save(path, trajectory)
        dlg.set_on_done(save_file)
        self.window.show_dialog(dlg)

    def load(self, path):
        gemoetry = o3d.io.read_triangle_model(path)
        if gemoetry is not None:
            short_name = os.path.basename(os.path.normpath(path))
            if short_name in self.geom_names:
                short_name = short_name + "_"
            self._scene.scene.add_model(short_name, gemoetry)
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60.0, bounds, bounds.get_center())

            # rescale for fitting the camera view
            em = self.get_camera_extrinsic_matrix()
            self._scene.setup_camera(
                self.intrinsic,
                em,
                bounds
            )
            
            def on_check_geometry(checked):
                self._scene.scene.show_geometry(short_name, checked)
            geom = gui.Checkbox(short_name)
            geom.checked = True
            geom.set_on_checked(on_check_geometry)

            self._geometry_ctrls.add_fixed(self.separation_height)
            self._geometry_ctrls.add_child(geom)
            self.window.set_needs_layout()

def main(args):
    gui.Application.instance.initialize()

    w = App(2048, 1536)
    if (args.inputpath != None):
        w.load(args.inputpath)
    if (args.trajpath != None):
        w.replay_camera_trajectory(np.load(args.trajpath))
    gui.Application.instance.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Viewer')
    parser.add_argument('-i', '--inputpath', default=None, type=str,
                        help='path to the obj file')        
    parser.add_argument('-t', '--trajpath', default=None, type=str,
                        help='path to the trajectory (.npy) file')        
    main(parser.parse_args())
