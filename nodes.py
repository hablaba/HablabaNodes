from random import random
import cv2

from retinaface import RetinaFace
from PIL import Image

import numpy as np
from HablabaNodes.comfy_annotations import (
    NumberInput,
    ComfyFunc,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)

import torch
import torchvision

my_category = "Comfy Annotation Examples"


# This is the converted example node from ComfyUI's example_node.py.example file.
@ComfyFunc(my_category)
def crop_face(
    image: ImageTensor,
    padding: int = NumberInput(0, 0, 4096, 1, "number"),
) -> ImageTensor:
    
    # print("hi!")
    
    crops = []
    for _image in image:
        i = 255. * _image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
        cv2_image = np.array(img)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        
        face = RetinaFace.detect_faces(cv2_image)["face_1"]

        x, y, x2, y2 = face["facial_area"]
        w = x2 - x
        h = y2 - y
        
        yc = int(y + (h / 2))
        xc = int(x + (w / 2))
        
        max_dim = int((max([w,h])/2)) + padding
        
        x = int(xc - max_dim)
        x2 = int(xc + max_dim)
        y = int(yc - max_dim)
        y2 = int(yc + max_dim)
        
        print(x,x2,y,y2)
            
        cropped = cv2_image[y:y2, x:x2]
        
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped = Image.fromarray(cropped)
        cropped = cropped.convert("RGB")
        cropped = np.array(cropped).astype(np.float32) / 255.0
        cropped = torch.from_numpy(cropped)[None,]
        
        crops.append(cropped)
    
    # print("hello?")
    return torch.cat(crops, dim=0)
