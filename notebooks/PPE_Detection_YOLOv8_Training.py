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
