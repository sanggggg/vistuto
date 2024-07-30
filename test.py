from PyQt5 import QtWidgets, QtGui,QtCore
import open3d as o3d
from subprocess import Popen, PIPE
# import win32gui
import sys

from OpenGL.arrays import vbo
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        self.setCentralWidget(widget)

        pcd = o3d.io.read_point_cloud("example.pcd")
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(pcd)

        # hwnd = win32gui.FindWindowEx(0, 0, None, "Open3D")
        hwnd = self.get_hwnd()
        self.window = QtGui.QWindow.fromWinId(hwnd)    
        self.windowcontainer = self.createWindowContainer(self.window, widget)
        layout.addWidget(self.windowcontainer, 0, 0)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(1)

    def get_hwnd(self):
        hwnd = None
        while hwnd == None:
            proc = Popen('wmctrl -l', stdin=None, stdout=PIPE, stderr=None, shell=True)
            out, err = proc.communicate()
            for window in out.decode('utf-8').split('\n'):
                if 'Open3D' in window:
                    hwnd = int(window.split(' ')[0], 16)
                    return hwnd

    def update_vis(self):
        # self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.setWindowTitle('o3d Embed')
    form.setGeometry(100, 100, 600, 500)
    form.show()
    sys.exit(app.exec_())
