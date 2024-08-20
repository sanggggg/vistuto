# Simple Mesh Viewer

3D mesh viewer, can hide and show multiple meshes, can record pinhole camera trajectories and replay saved trajectories.

```sh 
python simple_viewer.py
python simple_viewer.py -i $(ASSET_PATH)
```

# FLAME Viewer

weight files are from https://flame.is.tue.mpg.de
FLAME face model viewer, realtime modifying parameter to render facial mesh. Can put sunglasses on facial mesh.
Weights files :

```sh
python flame_viewer.py
```

# DECA viewer

weight files are from https://deca.is.tue.mpg.de

```sh
python deca_viewer.py -i $(ASSET_VIDEO_PATH)
python deca_viewer.py -i $(ASSET_VIDEO_PATH) --cool # put sunglasses on face
```

# DECA Server

```sh
gunicorn service:app --workers 4 --reload --port 1234 --host 0.0.0.0
```
