# PPE Detection Training Script
# I usually run this in colab so the paths reflect that

import torch
import os
import yaml
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# print out if we actually got a GPU
print(f"using torch {torch.__version__}, GPU: {torch.cuda.is_available()}")

from roboflow import Roboflow

# grab the dataset
# TODO: move the api key to an env var so I don't accidentally leak it on github again
try:
    rf = Roboflow(api_key="TIkrECrV61SUEZuRwnih") 
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    version = project.version(30)
    dataset = version.download("yolov8")
    print("dataset downloaded to:", dataset.location)
    dataset_dir = dataset.location
except Exception as e:
    print("roboflow broke again...", e)
    dataset_dir = "./dataset" # fallback

yaml_path = os.path.join(dataset_dir, "data.yaml")

# just double check what we downloaded
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)
    print(f"found {config['nc']} classes: {config['names']}")

# mount drive so we can save the big weights file at the end
try:
    from google.colab import drive
    drive.mount('/content/drive')
    drive_dir = '/content/drive/MyDrive/YOLOv8_PPE_Detection'
    os.makedirs(drive_dir, exist_ok=True)
except:
    print("not in colab, skipping drive mount")
    drive_dir = './saved_weights'

local_run_dir = '/content/runs/detect' if 'colab' in str(drive_dir) else './runs/detect'

# --- THE ACTUAL TRAINING ---
from ultralytics import YOLO

# load medium model. large is too slow for webcam inference later
model = YOLO('yolov8m.pt')

print("starting training... grab a coffee this takes a while")
# I lowered batch size to 16 cause it kept OOMing on the free colab T4s
res = model.train(
    data=yaml_path,
    epochs=100, # 100 should be enough
    imgsz=640,
    batch=16, 
    patience=20, # stop early if it sucks
    optimizer='AdamW',
    lr0=0.001,
    save_period=10, 
    project=local_run_dir,
    name='ppe-yolov8m',
    pretrained=True
)

print("done training!!")

# validation
print("running val...")
metrics = model.val()
print("mAP:", metrics.box.map)

# save out the best.pt so we don't lose it when colab disconnects
import shutil
run_path = os.path.join(local_run_dir, 'ppe-yolov8m')
best_pt = os.path.join(run_path, 'weights', 'best.pt')

if os.path.exists(best_pt):
    out = os.path.join(drive_dir, 'best.pt')
    shutil.copy2(best_pt, out)
    print("saved weights to", out)
else:
    print("uh oh, couldn't find best.pt!")

# export to onnx just in case we need it later
print("exporting onnx...")
model.export(format='onnx', opset=12)
print("all done :)")
