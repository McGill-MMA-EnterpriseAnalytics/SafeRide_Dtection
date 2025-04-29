from ultralytics import YOLO
import os

def load_number_plate_model(model_path="models/number_plate_detection_best.pt"):
    """
    Loads the YOLOv8 model for number plate detection.

    Args:
        model_path (str): Path to the .pt file

    Returns:
        YOLO model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return YOLO(model_path)

def detect_number_plates(model, image_path, conf=0.3):
    """
    Detects number plates in the input image.

    Args:
        model: Loaded YOLO model
        image_path (str): Path to input image
        conf (float): Confidence threshold

    Returns:
        List of bounding boxes. Each bounding box is [x1, y1, x2, y2]
    """
    results = model.predict(source=image_path, conf=conf)
    
    bboxes = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy:   # xyxy means [x1, y1, x2, y2]
            bboxes.append(box.tolist())
    
    return bboxes
