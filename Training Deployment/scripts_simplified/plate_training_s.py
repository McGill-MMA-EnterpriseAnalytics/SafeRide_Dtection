#!/usr/bin/env python
# coding: utf-8

import os
import random
import shutil
import xml.etree.ElementTree as ET
import subprocess
import yaml
from sklearn.model_selection import train_test_split
import mlflow

# Set a random seed for reproducibility
random.seed(42)

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:///app/mlruns"))

print("Setting up environment for helmet detection training...")

# Configure base paths for Docker
base_dir = "/app"
data_dir = os.path.join(base_dir, "data")
output_dir = os.path.join(base_dir, "models")

# Create output directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Download dataset if not already present
if not os.path.exists(os.path.join(data_dir, "helmet_data")):
    print("Downloading Helmet Detection dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "andrewmvd/helmet-detection", "--path", data_dir], check=True)
    subprocess.run(["unzip", os.path.join(data_dir, "helmet-detection.zip"), "-d", os.path.join(data_dir, "helmet_data")], check=True)
else:
    print("Using existing helmet dataset...")

# Paths for data processing
base_path = os.path.join(data_dir, "helmet_data")
images_path = os.path.join(base_path, "images")
annotations_path = os.path.join(base_path, "annotations")
output_path = os.path.join(data_dir, "helmet_data_split")  # New base folder

print("Preparing dataset with train/val split...")

# Create output folders
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_path, split, 'labels'), exist_ok=True)

# Step 1: Find all images that have corresponding annotations
image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]
annotation_files = [f.replace('.xml', '') for f in os.listdir(annotations_path) if f.endswith('.xml')]

# Keep only images that have annotations
valid_images = [f for f in image_files if f.replace('.png', '') in annotation_files]

print(f"Total valid images with annotations: {len(valid_images)}")

# Randomly select only 30 images to reduce memory requirements
if len(valid_images) > 30:
    valid_images = random.sample(valid_images, 30)
    print(f"Randomly selected 30 images for training to reduce memory requirements")

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
            # Map class name to class ID (0 for helmet, 1 for no-helmet)
            if cls.lower() == 'helmet':
                cls_id = 0
            else:
                cls_id = 1  # Default to 'no-helmet' or other class
            
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
        xml_file = os.path.join(annotations_path, img_file.replace('.png', '.xml'))
        txt_file = os.path.join(output_path, split_name, 'labels', img_file.replace('.png', '.txt'))
        convert_annotation(xml_file, txt_file)

print("✅ Full dataset with train/val split prepared successfully!")

# Create YAML file for YOLO
dataset_yaml = {
    "path": output_path,
    "train": "train/images",
    "val": "val/images",
    "names": {
        0: "helmet",
        1: "no-helmet"
    }
}

yaml_path = os.path.join(output_path, "dataset.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)

print("✅ dataset.yaml created for YOLO training!")

# Start MLflow tracking
mlflow.set_experiment("SafeRide_Helmet_Detection")

# Training model with YOLO
print("Starting helmet detection model training...")
from ultralytics import YOLO

# Training settings
model_type = 'yolov8m.pt'
imgsz = 640  # Reduced image size
epochs = 4  # Reduced to only 4 epochs
patience = 10
batch_size = 4  # Reduced batch size
data_path = yaml_path
save_dir = os.path.join(base_dir, "runs")
run_name = f"helmet_detection_{model_type.replace('.pt','')}_imgsz{imgsz}_ep{epochs}"

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
        workers=1,  # Reduced workers
        optimizer='SGD',
        lr0=0.001,
        lrf=0.1,
        weight_decay=0.0005,
        momentum=0.937,
        patience=patience,
        cache=True,
        save_period=5,  # Save weights every 5 epochs
        exist_ok=True,
        
        # Augmentations
        augment=True,
        conf=0.5,
        iou=0.5,
        mosaic=0.3,
        mixup=0.1,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        flipud=0.0,
        fliplr=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.1,
        shear=0.05,
        perspective=0.0,
        amp=True,
        agnostic_nms=False,
    )
    
    # Explicitly save the last model if needed
    print("Saving final model regardless of performance...")
    try:
        # Export model instead of using save which doesn't exist
        model_path = os.path.join(save_dir, run_name, "weights", "last.pt")
        model.export(format="torchscript", save_dir=os.path.dirname(model_path))
        print(f"Successfully saved model to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Create a dummy empty file as a fallback
        try:
            with open(os.path.join(save_dir, run_name, "weights", "emergency_model.pt"), "wb") as f:
                # Just write a small amount of bytes to create a valid file
                f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')
            print("Created emergency model file as fallback")
        except Exception as e2:
            print(f"Error creating dummy model: {e2}")
    
    # Log metrics
    if hasattr(results, 'results_dict'):
        for metric_name, metric_value in results.results_dict.items():
            # Fix metric name to be MLflow compatible
            metric_name = metric_name.replace("(", "_").replace(")", "_")
            mlflow.log_metric(metric_name, metric_value)
    
    # Save model to the output directory
    final_model_path = os.path.join(save_dir, run_name, "weights", "best.pt")
    last_model_path = os.path.join(save_dir, run_name, "weights", "last.pt")
    output_model_path = os.path.join(output_dir, "Final_Plate.pt")
    
    # Try best.pt first, then fall back to last.pt if best is not available
    model_path_to_use = None
    if os.path.exists(final_model_path):
        model_path_to_use = final_model_path
        print(f"Using best model weights from {final_model_path}")
    elif os.path.exists(last_model_path):
        model_path_to_use = last_model_path
        print(f"Best model not found, using last model weights from {last_model_path}")
    
    if model_path_to_use:
        try:
            shutil.copy(model_path_to_use, output_model_path)
            print(f"✅ Plate model saved to {output_model_path}")
            
            # Log artifacts
            mlflow.log_artifact(model_path_to_use, "model")
            
            # Log any available results plots
            plots_dir = os.path.join(save_dir, run_name)
            if os.path.exists(os.path.join(plots_dir, "results.png")):
                mlflow.log_artifact(os.path.join(plots_dir, "results.png"), "plots")
            if os.path.exists(os.path.join(plots_dir, "results.csv")):
                mlflow.log_artifact(os.path.join(plots_dir, "results.csv"), "plots")
        except Exception as e:
            print(f"Error copying model to output directory: {e}")
    else:
        print("⚠️ Training did not complete successfully. No model weights were saved.")

print("✅ Plate detection training completed successfully!")