import cv2
from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import io
from PIL import Image
from src.decalib.deca import DECA
from src.decalib.utils.config import cfg as deca_cfg
from src.decalib.datasets import datasets

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
deca = DECA(config=deca_cfg, device='cuda')


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/meshes/image")
async def detect_objects(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    preprocessor = datasets.Preprocessor(iscrop=True, crop_size=224, scale=1.25, face_detector='fan')
    img = preprocessor.process(image)

    images = torch.stack([img['image']]).to('cuda')
    original_images = torch.stack([img['original_image']]).to('cuda')
    tforms = torch.stack([torch.inverse(img['tform']).transpose(0,1)]).to('cuda')
    with torch.no_grad():
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict, render_orig=True, tform=tforms, original_image=original_images)

    meshed_image = tensor2image(visdict['shape_images'][0])
    meshed_image = Image.fromarray(meshed_image)
    byte_array = io.BytesIO()
    meshed_image.save(byte_array, format='PNG')
    byte_data = byte_array.getvalue()
    return Response(content=byte_data, media_type="image/png")

# @app.post("/meshes/video")
# async def detect_objects(file: UploadFile):



def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()
