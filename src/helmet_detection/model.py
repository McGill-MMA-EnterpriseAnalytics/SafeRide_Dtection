from ultralytics import YOLO
import os

def load_helmet_model(model_path="models/helmet_detection_best.pt"):
    """
    Loads the YOLOv8 helmet detection model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

def is_helmet_detected(model, image_path, conf=0.3):
    """
    Returns True if at least one helmet is detected above the confidence threshold.
    
    Args:
        model: Loaded YOLO model
        image_path (str): Path to image
        conf (float): Confidence threshold

    Returns:
        bool: True if helmet detected, else False
    """
    results = model.predict(source=image_path, conf=conf)
    boxes = results[0].boxes
    num_detections = len(boxes) if boxes is not None else 0
    return num_detections > 0
