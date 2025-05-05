#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import xml.etree.ElementTree as ET
import yaml
from sklearn.model_selection import train_test_split
import mlflow
from plate_drift import detect_drift  # Drift detection

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns"))

print("Setting up environment for plate detection training...")

# Paths
base_dir = "/app"
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "models")
test_dir = os.path.join(data_dir, "test_data/plates/images")  # Using local test data
reference_dir = os.path.join(data_dir, "plate_data/images")

# Drift check
if not detect_drift(reference_dir, test_dir):
    print("No drift detected. Skipping training.")
    exit(0)

# Prepare split
output_path = os.path.join(data_dir, "plate_data_split")
os.makedirs(output_path, exist_ok=True)
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

# Split test data
image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg')]
train_imgs, val_imgs = train_test_split(image_files, test_size=0.2, random_state=42)
splits = {'train': train_imgs, 'val': val_imgs}

# Convert annotations from test_data (dummy placeholders assumed)
def convert_annotation_dummy(image_file, label_dir):
    open(os.path.join(label_dir, image_file.replace('.jpeg', '.txt')), 'w').close()

for split_name, img_list in splits.items():
    for img_file in img_list:
        src = os.path.join(test_dir, img_file)
        dst = os.path.join(output_path, split_name, 'images', img_file)
        shutil.copy(src, dst)
        convert_annotation_dummy(img_file, os.path.join(output_path, split_name, 'labels'))

# YAML for YOLO
yaml_path = os.path.join(output_path, "dataset.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump({
        "path": output_path,
        "train": "train/images",
        "val": "val/images",
        "names": {0: "plate"}
    }, f)

# Train using fine-tuning
from ultralytics import YOLO
mlflow.set_experiment("SafeRide_Plate_Detection")
model = YOLO(os.path.join(output_dir, "Final_Plates.pt")) if os.path.exists(os.path.join(output_dir, "Final_Plates.pt")) else YOLO("yolov8s.pt")

with mlflow.start_run(run_name="plate_finetune"):
    model.train(data=yaml_path, epochs=20, imgsz=960, batch=16, device=0, save=True, project="/app/runs", name="plate_finetuned", exist_ok=True)
    model_path = "/app/runs/plate_finetuned/weights/best.pt"
    shutil.copy(model_path, os.path.join(output_dir, "Final_Plates.pt"))
    mlflow.log_artifact(model_path, "model")