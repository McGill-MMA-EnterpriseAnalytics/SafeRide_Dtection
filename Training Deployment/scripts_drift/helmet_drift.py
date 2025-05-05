import os
import cv2
import numpy as np
from ultralytics import YOLO

# Threshold for relative change to flag drift (e.g., 40% change is considered significant)
DRIFT_THRESHOLD = 0.4

def compute_image_metrics(image_path, model=None):
    """
    Compute image-level visual metrics and scene complexity (vehicle count).
    - Brightness: mean intensity
    - Contrast: standard deviation of intensity
    - Sharpness: variance of Laplacian (edge detail)
    - Vehicle count: YOLO-predicted objects from common vehicle classes
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # If a YOLO model is provided, detect vehicles (car=2, motorcycle=3, bus=5, truck=7)
    vehicle_count = 0
    if model:
        results = model(image)
        vehicle_count = sum(1 for cls_id in results[0].boxes.cls if int(cls_id) in [2, 3, 5, 7])

    return brightness, contrast, sharpness, vehicle_count

def aggregate_folder_metrics(folder_path, model=None):
    """
    Compute average metrics across all valid images in a folder.
    """
    metrics = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file)
            metrics.append(compute_image_metrics(image_path, model))
    
    # Return mean across all collected metric tuples
    return np.mean(metrics, axis=0)

def detect_drift(reference_dir, test_dir, threshold=DRIFT_THRESHOLD):
    """
    Compare aggregated metrics between reference and test datasets.
    Flags drift if any relative metric change exceeds the threshold.
    """
    model = YOLO("yolov8n.pt")  # Lightweight YOLOv8 model for counting vehicles

    # Aggregate metrics from training data and new/test data
    ref_metrics = aggregate_folder_metrics(reference_dir, model)
    test_metrics = aggregate_folder_metrics(test_dir, model)

    metric_names = ['brightness', 'contrast', 'sharpness', 'vehicle_count']
    drift_flags = []

    print("\n📊 Drift Analysis Report:")
    for name, r, t in zip(metric_names, ref_metrics, test_metrics):
        rel_change = abs(t - r) / max(r, 1e-5)
        drift = rel_change > threshold

        print(f"  - {name}: ref={r:.2f}, test={t:.2f}, Δ={rel_change:.2%} → {'⚠️ Drift' if drift else '✅ OK'}")
        drift_flags.append(drift)

    # If any metric shows drift, return True
    return any(drift_flags)
