import os
import cv2
import numpy as np

# Threshold for relative change to flag drift (e.g., 40% or more)
DRIFT_THRESHOLD = 0.4

def compute_image_metrics(image_path):
    """
    Compute basic image-level quality metrics:
    - Brightness: mean pixel intensity
    - Contrast: standard deviation (spread of intensity)
    - Sharpness: Laplacian variance (edge focus)
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return brightness, contrast, sharpness

def aggregate_folder_metrics(folder_path):
    """
    Iterate through images in a folder and compute average metrics.
    Only uses images with valid extensions.
    """
    metrics = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file)
            metrics.append(compute_image_metrics(image_path))
    
    # Return mean metric vector: [brightness, contrast, sharpness]
    return np.mean(metrics, axis=0)

def detect_drift(reference_dir, test_dir, threshold=DRIFT_THRESHOLD):
    """
    Compare test dataset metrics to reference dataset (training).
    Return True if any metric deviates beyond the threshold.
    """
    ref_metrics = aggregate_folder_metrics(reference_dir)
    test_metrics = aggregate_folder_metrics(test_dir)

    metric_names = ['brightness', 'contrast', 'sharpness']
    drift_flags = []

    print("\n📊 Plate Drift Detection Report:")
    for name, r, t in zip(metric_names, ref_metrics, test_metrics):
        rel_change = abs(t - r) / max(r, 1e-5)
        drift = rel_change > threshold

        print(f"  - {name}: ref={r:.2f}, test={t:.2f}, Δ={rel_change:.2%} → {'⚠️ Drift' if drift else '✅ OK'}")
        drift_flags.append(drift)

    return any(drift_flags)
