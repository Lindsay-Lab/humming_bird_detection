# check_env.py
import torch
import torchvision
import numpy as np
import scipy
import cv2

from ultralytics import YOLO
from roboflow import Roboflow

print("=== Torch / CUDA ===")
print("torch:", torch.__version__)
print("compiled with CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.matmul(x, x)
    print("matmul ok, y mean:", y.mean().item())

print("torchvision:", torchvision.__version__)
print("numpy:", np.__version__)
print("scipy:", scipy.__version__)
print("opencv:", cv2.__version__)

import ultralytics
print("ultralytics:", ultralytics.__version__)

print("\n=== Roboflow ===")
_ = Roboflow
print("roboflow import OK")