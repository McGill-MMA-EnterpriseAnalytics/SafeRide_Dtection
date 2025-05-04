#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import xml.etree.ElementTree as ET
import subprocess
import yaml
from sklearn.model_selection import train_test_split
import mlflow
import random
# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns"))

print("Setting up environment for plate detection training...")

# Configure base paths for Docker
base_dir = "/app"
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "models")

# Create output directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Download dataset if not already present
if not os.path.exists(os.path.join(data_dir, "plate_data")):
    print("Downloading License Plate Detection dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "aslanahmedov/number-plate-detection", "--path", data_dir], check=True)
    subprocess.run(["unzip", os.path.join(data_dir, "number-plate-detection.zip"), "-d", os.path.join(data_dir, "plate_data")], check=True)
else:
    print("Using existing plate dataset...")

# Paths for data processing
base_path = os.path.join(data_dir, "plate_data")
images_path = os.path.join(base_path, "images")
annotations_path = os.path.join(base_path, "images")  # Annotations are in same dir as images
output_path = os.path.join(data_dir, "plate_data_split")  # New base folder

print("Preparing plate dataset with train/val split...")

# Create output folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

# Step 1: Find all images that have corresponding annotations
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpeg')]
annotation_files = [f.replace('.xml', '') for f in os.listdir(annotations_path) if f.endswith('.xml')]

# Keep only images that have annotations
valid_images = [f for f in image_files if f.replace('.jpeg', '') in annotation_files]

print(f"Total valid images with annotations: {len(valid_images)}")

# Randomly select only 30 images to reduce memory requirements

sample_size = 10
if len(valid_images) > sample_size:
    valid_images = random.sample(valid_images, sample_size)
    print(f"Randomly selected {sample_size} images for training to reduce memory requirements")
    
# Step 2: Split into train and val
train_imgs, val_imgs = train_test_split(valid_images, test_size=0.2, random_state=42)  # 80% train, 20% val
splits = {'train': train_imgs, 'val': val_imgs}

# Helper: XML to YOLO TXT converter
def convert_annotation(xml_file, txt_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            # Assuming 'plate' is class 0
            cls_id = 0
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymin = int(xmlbox.find('ymin').text)
            ymax = int(xmlbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height

            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Step 3: Move and convert
for split_name, img_list in splits.items():
    for img_file in img_list:
        img_src = os.path.join(images_path, img_file)
        img_dst = os.path.join(output_path, split_name, 'images', img_file)
        shutil.copy(img_src, img_dst)

        # Convert annotation
        xml_file = os.path.join(annotations_path, img_file.replace('.jpeg', '.xml'))
        txt_file = os.path.join(output_path, split_name, 'labels', img_file.replace('.jpeg', '.txt'))
        convert_annotation(xml_file, txt_file)

print("✅ Full plate dataset with train/val split prepared successfully!")

# Create YAML file for YOLO
dataset_yaml = {
    "path": output_path,
    "train": "train/images",
    "val": "val/images",
    "names": {
        0: "plate"
    }
}

yaml_path = os.path.join(output_path, "dataset.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)

print("✅ dataset.yaml created for YOLO plate detection training!")

# Start MLflow tracking
mlflow.set_experiment("SafeRide_Plate_Detection")

# Training model with YOLO
print("Starting plate detection model training...")
from ultralytics import YOLO

# Training settings
model_type = 'yolov8s.pt'
imgsz = 640  # Reduced image size
epochs = 2  # Reduced to only 2 epochs
patience = 10
batch_size = 4  # Reduced batch size
data_path = yaml_path
save_dir = os.path.join(base_dir, "runs")
run_name = f"plate_detection_{model_type.replace('.pt','')}_imgsz{imgsz}_ep{epochs}"

# Start MLflow run
with mlflow.start_run(run_name=run_name):
    # Log parameters
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("imgsz", imgsz)
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("patience", patience)
    mlflow.log_param("batch_size", batch_size)
    
    # Initialize model
    model = YOLO(model_type)
    
    # Train model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        save=True,
        project=save_dir,
        name=run_name,
        batch=batch_size,
        device='cpu',
        workers=1,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.01,
        momentum=0.9,
        patience=patience,
        close_mosaic=20,
        cache=True,
        save_period=5,
        exist_ok=True,
        
        # Augmentations
        augment=True,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        flipud=0.0,
        fliplr=0.1,
        degrees=5.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.005,
        amp=True,
        agnostic_nms=False,
    )
    
    # Log metrics
# Log metrics
if hasattr(results, 'results_dict'):
    for metric_name, metric_value in results.results_dict.items():
        # Clean up metric names to follow MLflow naming rules
        clean_metric_name = metric_name.replace("(", "_").replace(")", "_")
        try:
            mlflow.log_metric(clean_metric_name, metric_value)
        except Exception as e:
            print(f"Warning: Could not log metric {clean_metric_name}: {e}")
    
    # Save model to the output directory
    final_model_path = os.path.join(save_dir, run_name, "weights", "best.pt")
    output_model_path = os.path.join(output_dir, "Final_Plates.pt")
    
    shutil.copy(final_model_path, output_model_path)
    print(f"✅ Plate detection model saved to {output_model_path}")
    
    # Log artifacts
    mlflow.log_artifact(final_model_path, "model")
    
    # Log any available results plots
    plots_dir = os.path.join(save_dir, run_name)
    if os.path.exists(os.path.join(plots_dir, "results.png")):
        mlflow.log_artifact(os.path.join(plots_dir, "results.png"), "plots")
    if os.path.exists(os.path.join(plots_dir, "results.csv")):
        mlflow.log_artifact(os.path.join(plots_dir, "results.csv"), "plots")

print("✅ Plate detection training completed successfully!")